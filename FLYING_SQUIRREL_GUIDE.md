# Flying-Squirrel Glider — Complete Guide & Plan

End-to-end reference for bringing a parametrically generated flying-squirrel mesh into the PFM-SFI 3D solver, running it in flight, visualizing results, and iterating toward an RL-controlled tree-to-tree glider.

This is the single source of truth for the squirrel-glider work. Prior stand-alone docs (MESH_REQUIREMENTS, GLIDER_SETUP_GUIDE, PARAMETRIC_MESH_ARCHITECTURE, PARAMETRIC_MESH_QUICKSTART, GETTING_STARTED) were folded in here and deleted — check `git log` if you need the old text.

---

## 0. Goal & system at a glance

**Long-term**: RL-controlled deformable glider jumping tree-to-tree in a fluid domain. Agent commands the wing's control anchors; solver resolves aerodynamic loads on a deformable patagium.

**Near-term** (this guide): Stable wind-tunnel simulation of a parametrically generated squirrel mesh. Prove the pipeline, tune materials, then climb the complexity ladder.

**Pipeline**:
```
parametric_mesh/config/*.yaml    ← bone lengths, angles, appendages, camber
          │
          ▼
parametric_mesh/scripts/mesh_generator.py  (runs inside Blender)
   builds armature → boundary curve → Grid Fill → camber → triangulate → export
          │
          ▼
3D/assets/mesh/squirrel.obj    (+ optional squirrel_metadata.json)
          │
          ▼
3D/configs/squirrel.yaml    ← material, fluid scenario, fixed_region (torso pin)
          │
          ▼
3D/run.py   (IBM_CLOTH_MODE=yaml IBM_CLOTH_CONFIG=...)
   TrianMesh normalize → XPBD constraints → IBM coupling to Eulerian fluid
          │
          ▼
logs/<exp_name>/    .ply solid frames + .vti fluid fields + 2D slice jpgs
          │
          ▼
Paraview animation     (later: RL loop writes control targets per step)
```

---

## Part 1 — The mesh

### 1.1 Shape: single-layer silhouette, not a solid

The mesh is **one connected triangulated surface** shaped like the top-view silhouette of a spread squirrel — head, torso strip, propatagium, main patagium, uropatagium, tail — all joined as one membrane. No thickness. No separate body volume.

Why:
- `TrianMesh.get_mass_3d` at `3D/gmesh.py:112` derives per-vertex mass from triangle **area** × `rho`. It has no concept of volume; a cube would be under-massed.
- `Bend3D` at `3D/bend.py` computes dihedral angles between adjacent face pairs. On a single-layer membrane, every interior edge has two incident faces (bend works); the silhouette boundary has one-face edges (boundary, fine).
- IBM coupling (`sample_ibm_u`, `spread_force` in `3D/ibm_cloth.py`) is symmetric across the surface — both sides of a membrane see the fluid.
- The existing working example, `3D/assets/mesh/silk2.obj`, is exactly this: a plane subdivision. Proven codepath.
- Physically the patagium is ~1 mm of skin; adding thickness buys nothing at grid scales of `1/128` or `1/256`.

Note: earlier docs in this repo called for a "closed, watertight" mesh — that was overstated, carried over from the parachute example. The silk flag violates it and the sim runs fine.

### 1.2 Topology target

```
             head (point)
              /\
         ____/  \____
        /            \       ← propatagium strip (neck ↔ wrist_L/R)
   ____/              \____
  /       torso strip      \  ← this central band gets the fixed_region pin
 /__________________________\
  \__     ___________     __/
     \   /           \   /   ← main patagium (wrist ↔ ankle each side)
      \_/             \_/
       |               |     ← uropatagium (ankle ↔ tail)
        \             /
         \___________/
              tail (point)
```

Vertex-density rule of thumb:
- **Dense** at body↔patagium transition (high curvature region; coarse tris kink).
- **Dense** at patagium trailing edge (vortex shedding, want resolution).
- **Dense** at styliform process attachments (sharp angle, concentrates deformation).
- **Sparse** in flat mid-wing regions.

Triangle budgets:
- Stage 0 shakedown: 2–4k
- Stage 1 material tune: 4–8k
- Stage 2 production: 8–15k

### 1.3 Blender requirements

From `3D/gmesh.py:30` via `meshio.read(...).cells_dict['triangle']`:

- **Triangulated only** — quads/n-gons → `KeyError` at load.
- **Consistent outward normals** — doesn't change IBM force coupling (symmetric) but keeps bend-constraint signs consistent and colors correctly in Paraview.
- **No duplicate vertices** — Merge by Distance before export.
- **No loose geometry** — Clean Up → Delete Loose.
- **Orientation**: +X forward (flight direction), +Z up, origin at torso centroid. `TrianMesh` recenters by centroid and rescales to unit bbox (`gmesh.py:45-52`), so Blender units don't matter — proportions and orientation do.

OBJ export settings:
- Selection Only, Apply Modifiers, **Triangulated Mesh ON**, forward=X, up=Z, scale=1.0.

Preflight validator (run from `3D/`):
```python
import meshio, numpy as np
m = meshio.read("assets/mesh/squirrel.obj")
assert 'triangle' in m.cells_dict, "not triangulated"
f = m.cells_dict['triangle']
edges = np.sort(np.vstack([f[:,[0,1]], f[:,[1,2]], f[:,[2,0]]]), 1)
uniq, counts = np.unique(edges, axis=0, return_counts=True)
boundary = (counts == 1).sum()
interior = (counts == 2).sum()
broken = (counts > 2).sum()
print(f"verts={len(m.points)} tris={len(f)} boundary_edges={boundary} interior_edges={interior} broken_edges={broken}")
```
Expected for a single-layer silhouette: `broken_edges=0`, `boundary_edges` equal to the silhouette outline edge count (nonzero, several hundred for a reasonable mesh).

### 1.4 Mesh generator — extend `parametric_mesh/scripts/mesh_generator.py`

**Chosen approach: parameterized Blender Python script.** Rationale: reproducible from config, version-controllable as text, parameter-sweepable via Hydra multirun, directly mutable by future RL agent, no binary artifacts to maintain.

Existing scaffolding:
- `parametric_mesh/scripts/mesh_generator.py` — Hydra-driven, builds armature, applies bone angles, adds appendages (partial), generates patagium (stub).
- `parametric_mesh/config/armature/base_rig.yaml` — bone length proportions, spread angles, anchor points.
- `parametric_mesh/config/appendages/squirrel_standard.yaml` — `wrist_styloid`, `elbow_spur`, `ankle_spur`, `tail_cartilage`, `neck_fold_support` with `enabled` flags.
- `parametric_mesh/config/species/flying_squirrel_minimal.yaml` — species-level patagium spline definitions.

Gaps to fill (in priority order):

| # | Function in `mesh_generator.py` | Current state | Needed |
|---|---------------------------------|---------------|--------|
| 1 | `create_membrane_from_spline` (l.325–395) | 4-vertex quad stub between two anchors | Real boundary curve → `bpy.ops.mesh.fill_grid()` → triangulate |
| 2 | `generate_body_mesh` (l.257–279) | Standalone cylinder placeholder | **Remove** — body is the central strip of the membrane, not a separate object |
| 3 | `add_appendages` (l.231–255) | Handles `wrist_styloid` only | Fan out to `elbow_spur`, `ankle_spur`, `tail_cartilage`, `neck_fold_support` |
| 4 | camber / dihedral displacement | Not present | New function applied after Grid Fill; see Part 2 |
| 5 | fixed-indices metadata export | Not present | Write torso/spine vertex indices to `_metadata.json` for `fixed_region.type: indices` path |

Unified mesh generation pipeline (target):

```
clear_scene()
create_armature()                   # existing — OK
apply_bone_transforms()             # existing — OK
add_appendages()                    # extend to all 5 appendage types
compute_anchor_positions()          # existing — OK
build_silhouette_boundary()         # NEW: walk anchors, emit Bezier boundary loop
fill_grid_patagium()                # NEW: bpy.ops.mesh.fill_grid inside loop
apply_camber()                      # NEW: Z-displacement as YAML function (Part 2)
apply_dihedral()                    # NEW: bend wing tips downward (Part 2)
mark_vertex_regions()               # body / membrane / styliform → vertex groups
triangulate_and_validate()          # existing — OK
export_obj_blend_metadata()         # extend metadata with fixed_indices
```

**Reference model (optional but helpful)**: a hand-authored .blend at one representative parameter set lets the generator's output be visually validated. Not source of truth; just QA target. Export the OBJ and drop it at `parametric_mesh/tests/fixtures/reference_squirrel.obj`.

**Running it** (from `parametric_mesh/`):
```bash
blender --background --python scripts/mesh_generator.py -- \
    --config-name=flying_squirrel_minimal
# with overrides:
blender --background --python scripts/mesh_generator.py -- \
    --config-name=flying_squirrel_minimal \
    armature.trunk_length=0.22 \
    patagium.camber.amplitude=0.05
# cross-platform via wrapper:
python scripts/blender_wrapper.py --script mesh_generator.py
```

---

## Part 2 — Curvature

Three independent layers. Address them in order as you climb the ladder.

### 2.1 Rest-pose curvature (camber + static dihedral)

The OBJ geometry **is** the rest pose. `TrianMesh.__init__` copies `v_p` into `v_p_ref` (`gmesh.py:60-61`). `LengthCons` and `Bend3D` measure deformation as deviation from `v_p_ref`. Whatever shape the .obj has is the shape the mesh wants to return to.

Put camber here. Added to the generator script as a Z-displacement function applied after Grid Fill:

```python
# in mesh_generator.py, new method apply_camber()
# s ∈ [-1, 1]  spanwise (wing root → wing tip)
# t ∈ [ 0, 1]  chordwise (leading edge → trailing edge)
z = camber_amp * sin(pi * t) + dihedral_amp * abs(s)
```

YAML schema additions (`config/species/flying_squirrel_minimal.yaml`):
```yaml
patagium:
  camber:
    amplitude: 0.04         # peak camber, fraction of chord
    profile: sinusoidal     # sinusoidal | naca4 | flat
  dihedral_angle: 5.0       # degrees; wing tips below body plane
```

Where `s` and `t` come from: the boundary curve walk gives you chord lines via the anchor skeleton (neck, wrist, ankle, tail). Parametrize the Grid Fill output by projecting each vertex onto the body axis (→ `s`) and the local chord direction (→ `t`).

### 2.2 Bone-pose curvature (two rest meshes)

Swap rest meshes for different flight phases. Same bones, different `bone_angles` and `camber.amplitude`. Easiest, most stable way to get "airplane vs parachute" without touching the solver.

Two sibling configs:

`config/species/flying_squirrel_cruise.yaml`:
```yaml
armature:
  bone_angles:
    forelimb_spread: 75.0
    hindlimb_spread: 80.0
    wing_droop: 0.0          # horizontal wing
patagium:
  camber:
    amplitude: 0.03          # light camber
  dihedral_angle: 2.0        # shallow
```

`config/species/flying_squirrel_landing.yaml`:
```yaml
armature:
  bone_angles:
    forelimb_spread: 60.0    # swept forward
    hindlimb_spread: 70.0
    wing_droop: -25.0        # wings angled down
patagium:
  camber:
    amplitude: 0.08          # deep camber (cupped)
  dihedral_angle: -20.0      # parachute shape: negative = tips below body
```

Run each as an independent simulation. Compare lift/drag from the wake vorticity structure.

### 2.3 Active curvature (the airplane↔parachute control signal)

This is the path to true glide-then-land dynamics and the eventual RL controller.

The mechanism is already half-built into `IBMClothConfig`:
- `cons_vert_i` (Taichi i32 field) — pinned vertex indices.
- `cons_pos` (numpy array) — target positions for those vertices.
- `solve_for_xpbd` at `3D/ibm_cloth.py:187-196` re-applies `cons_pos` every XPBD step via `cons_vert_p.from_numpy(cons_pos)` + `mesh.set_pos_by_index`.

So if anything mutates `cons_pos` between simulation steps, the next solve drives those vertices to the new targets.

Design for stage 3+:
1. Pin a small set of **control vertices** — typically 6: wrist_L, wrist_R, ankle_L, ankle_R, elbow_L, elbow_R. Write indices to `squirrel_metadata.json` under `control_vertices`.
2. Use `fixed_region.type: indices` in the YAML to select them as the fixed set (the infrastructure for this was added to `ibm_cloth_base.py` last session — see §4.1).
3. Add a small controller API to `IBMClothConfig`:
   ```python
   def set_control_targets(self, targets: np.ndarray):
       """targets shape: (n_control, 3). Updates cons_pos in place."""
       self.cons_pos[:] = targets
   ```
4. Call `cloth.set_control_targets(...)` from `run.py`'s main loop (frame or sub-step granularity) — fed by keyframe interpolation, a scripted policy, or eventually an RL agent.

Stiffness interplay:
- High `bend_constraint_alpha` → wing shape is mostly governed by anchor positions; membrane resists the fluid.
- Low `bend_constraint_alpha` → fluid pushes membrane freely between anchors; behaves more like a passive parachute canopy inflating.

For RL: action space = control target deltas; observation space = wake vorticity + body pose; reward = forward distance per unit altitude loss (glide ratio) with terminal bonus for landing within a target bbox.

---

## Part 3 — Simulation integration

### 3.1 The YAML config (3D/configs/squirrel.yaml)

Created in a prior session. Schema:

```yaml
metadata:
  name: squirrel            # expects assets/mesh/squirrel.obj

material_properties:
  global_scale: 0.35        # post-normalization size in domain
  repose_position: [0.25, 0.5, 0.5]
  regions:
    membrane:
      density: 1.5
      length_constraint_alpha: 0.02    # stretch compliance; ↑ = more elastic
      bend_constraint_alpha: 4000      # bend compliance; ↑ = floppier (parachute); ↓ = stiffer (airplane)

simulation:
  solve_iters: 50
  dt: 0.0005
  gravity: [0.0, 0.0, 0.0]  # wind-tunnel: off. Free-glide: e.g. [0, 0, -3]

constraints:
  fixed_region:
    type: bbox              # bbox | indices
    min: [0.22, 0.47, 0.47] # post-normalization domain coords
    max: [0.30, 0.53, 0.53] # torso band
```

Coordinates are **post-normalization domain frame**: `TrianMesh` recenters by centroid and rescales to unit bbox, THEN applies `global_scale` and translates by `repose_position`. Tune `repose_position` + `fixed_region` bbox using the stage-0 shakedown output.

### 3.2 Fixed-point selector precedence (`3D/ibm_cloth_base.py`)

Three-level fallback in `initialize_fixed_points`:

1. **`constraints.fixed_region` in YAML** — dispatches on `type`:
   - `type: bbox` → vertices inside `[min, max]` axis-aligned box.
   - `type: indices` → reads explicit index list from YAML `values:` field, or from `metadata['fixed_indices']` in the `<mesh>_metadata.json` file.
2. **`fixed_indices` in metadata.json** — explicit list written by the mesh generator (for RL hook, per-step control targets).
3. **Legacy spatial rule** `x < 0.205` — silk-flag compatibility, still used when `IBM_CLOTH_MODE=legacy`.

### 3.3 Fluid scenario (`scenario` in `3D/hyperparameters.py`)

Added knob near line 36:
```python
scenario = "vorts_oblique"   # default keeps prior behavior
inflow_U = 1.5               # used when scenario == "glide_wind"
```

`init_vorts()` at `3D/run.py:400-409` dispatches:
- `"glide_wind"` → `init_uniform_flow(X, u, smoke, tmp_smoke, inflow_U)` (uniform +X inflow, body pinned → wind tunnel).
- `"glide_free"` → still air (pass; relies on gravity + initial mesh velocity for a free-falling glider).
- else → existing `init_vorts_oblique`.

Boundary conditions `run.py:27` (`boundary_types = [[2,1],[1,1],[1,1]]`) put Neumann on x-min (inflow face), Dirichlet elsewhere — already correct for wind-tunnel inflow along +X. No change needed for the free-glide variant either.

### 3.4 Running the simulation

```bash
cd 3D
IBM_CLOTH_MODE=yaml IBM_CLOTH_CONFIG=configs/squirrel.yaml python run.py
```

Environment: conda env named `flowmap` (not "pfm-sfi" — see `environment.yml`), CUDA GPU required, stage-dependent memory (see §5).

---

## Part 4 — Visualization (Paraview offline)

Per-frame outputs land in `logs/<exp_name>/`:
- `solid/solid_{f:04d}.ply` — binary PLY of the animated mesh (written by `Export` in `ibm_cloth.py:86`).
- `vtks/field_{f:03d}.vti` — 3D VTK ImageData (vorticity magnitude + smoke scalars).
- `vort_2D/{f:03d}.jpg`, `smoke_2D/{f:03d}.jpg` — mid-z slices. Fast-look while the sim runs.
- `ckpts/*.npy` — velocity + particle checkpoints.

Write cadence is driven by `visualize_dt = 0.05` in hyperparameters; each visualize frame triggers `run.py:895-900` to write all of the above.

**Paraview recipe** (once per stage):
1. File → Open → `logs/<exp_name>/vtks/field_..vti` (Paraview auto-groups the time series).
2. File → Open → `logs/<exp_name>/solid/solid_..ply` (select all files, "Open as series").
3. On the `.vti` source: filter → **Contour** on `vorticity` (iso ≈ 1.0) for vortex tubes, or **Volume** render on `smoke` for dye-trace visuals.
4. On the `.ply` source: Representation = Surface, color by solid color.
5. Set animation range; Play.
6. File → Save Animation → mp4.

Save a `.pvsm` Paraview state file at `logs/paraview_squirrel_state.pvsm` once dialed in so later stages open in one click.

**Quick-look mp4 from 2D slices** (no Paraview needed):
```bash
ffmpeg -framerate 30 -i logs/<exp>/vort_2D/%03d.jpg -c:v libx264 -pix_fmt yuv420p logs/<exp>/vort.mp4
```

**Live viewer**: `TaichiRenderer3D` in `3D/utils/renderer.py` exists but is not wired into `run.py`'s main loop. Don't wire it up yet; it slows the sim and Paraview offline is better for this workflow.

---

## Part 5 — Staged iteration ladder

Change `exp_name` per stage — `logs/<exp_name>/` is overwritten in place.

| Stage | Purpose | `res_x, y, z` | `total_frames` | `solve_iters` | `device_memory_GB` (`run.py:12`) | `scenario` | Mesh complexity | Curvature (Part 2) |
|-------|---------|---------------|----------------|---------------|-----------------------------------|------------|-----------------|---------------------|
| **0 — shakedown** (minutes) | Mesh loads, doesn't NaN, torso pin catches vertices | 64, 32, 32 | 60 | 20 | 2.5 | `glide_wind` | 2–4k tris, flat planiform | none (flat) |
| **1 — material tune** (~1 h each) | Dial `length_constraint_alpha` × `bend_constraint_alpha` for stable patagium flutter | 128, 64, 64 | 200 | 30 | 5.0 | `glide_wind` | 4–8k tris | light camber (§2.1) |
| **2 — production** (overnight) | Full-quality baseline wind-tunnel animation | 256, 128, 128 | 1500 | 50 | 10.5 | `glide_wind` | 8–15k tris | camber + static dihedral |
| **3 — cruise vs landing** | Two rest meshes, compare aerodynamics | 256, 128, 128 | 1500 | 50 | 10.5 | `glide_wind` | 8–15k tris | two configs (§2.2) |
| **4 — free glide** | Body unpinned, gravity, initial velocity | 256, 128, 128 | 1500 | 50 | 10.5 | `glide_free` | 8–15k tris | rest-pose only |
| **5 — active control** | Controller mutates `cons_pos` each step; scripted then RL | 256, 128, 128 | variable | 50 | 10.5 | `glide_free` | 8–15k tris | active (§2.3) |

Stage go/no-go — only advance when the prior stage's verification checks pass. See Part 7.

Stage-knob edits all live in `3D/hyperparameters.py` (lines 24–26, 35–36, new lines for `scenario`/`inflow_U`), `3D/run.py:12` (`device_memory_GB`), and the YAML `material_properties.regions.membrane.{length_constraint_alpha, bend_constraint_alpha}`.

---

## Part 6 — Infrastructure changes already in place

Completed in prior sessions. No action needed — listed for reference.

### 6.1 `3D/ibm_cloth_base.py`
- Initialized `self.metadata = None` in `__init__` (fixes AttributeError in legacy mode).
- Rewrote `initialize_fixed_points` with the three-level precedence described in §3.2.
- Added `_fixed_indices_from_region(region)` handling `type: bbox` and `type: indices`.

### 6.2 `3D/init_conditions.py`
- Added `init_uniform_flow(X, u, smoke1, smoke2, U)` (plain Python) + `_fill_uniform_velocity` Taichi kernel, matching the existing wrapper pattern.

### 6.3 `3D/hyperparameters.py`
- Added `scenario = "vorts_oblique"` (default preserves prior behavior) and `inflow_U = 1.5`.

### 6.4 `3D/run.py`
- `init_vorts()` (lines ~400–409) now dispatches on `scenario`.

### 6.5 `3D/configs/squirrel.yaml`
- Created with the schema documented in §3.1.

### 6.6 `CLAUDE.md`
- Fixed-point note updated to reflect the new three-level precedence.

### 6.7 Not yet done
- Mesh generator script rewrite (§1.4).
- Camber/dihedral displacement functions (§2.1).
- Active-control API on `IBMClothConfig` (§2.3).
- Reference hand-authored Blender file (optional; from you).

---

## Part 7 — Verification

Run each check before advancing a stage.

### 7.1 Mesh preflight (before Stage 0)
- Meshio validator in §1.3 reports `broken_edges=0` and nonzero `boundary_edges` (single-layer membrane) or `boundary_edges=0` if you accidentally made a closed surface (fine too, but solver calibration will differ).
- Mesh opens in Blender with consistent face orientation (Viewport Overlays → Face Orientation: all blue).

### 7.2 Stage 0 shakedown
- Sim completes 60 frames without NaN.
- `logs/squirrel_s0/solid/solid_0000.ply` and `solid_0059.ply` both exist.
- Opening `solid_0000.ply` in Blender: pose/scale match your export; squirrel is roughly centered in the domain.
- Stdout contains `[IBMCloth] Fixed N vertices` with N > 0 and N ≲ 50 (torso band, not everything and not nothing).
- `logs/squirrel_s0/vort_2D/*.jpg` sequence grows as the run progresses — if it stalls for minutes, check Taichi console for NaN in the pressure solver.

### 7.3 Stage 1 material tune
- Sweep `(length_constraint_alpha, bend_constraint_alpha)` in `{0.005, 0.015, 0.03} × {2000, 4000, 8000}`.
- For each combination, produce a full 200-frame run with distinct `exp_name` (e.g. `squirrel_s1_aL0.015_aB4000`).
- In Paraview, scrub the `.ply` time series. Disqualify pairs with: edge flicker ("buzzing"), membrane inversion (flipping inside-out), or blow-up (extreme stretching then NaN).
- Lock the pair whose patagium flexes smoothly under the wake without any of the above.

### 7.4 Stage 2 production
- 1500 `.vti` + 1500 `.ply` files.
- Paraview animation exports cleanly to mp4.
- Wake structure looks physically plausible: streamwise tip vortices shedding off the wingtips, turbulent wake behind the body.
- Save the Paraview state file for reuse.

### 7.5 Stage 3 cruise-vs-landing
- Both rest meshes load and run at the same material parameters.
- Visible difference in wake vorticity strength and spread between the two configs.
- Qualitative: landing config shows stronger vortex shedding and broader wake (more drag).

### 7.6 Stage 4 free-glide readiness
- `scenario = "glide_free"` branch compiles (already verified by AST; verify at runtime).
- Removing `fixed_region` from the YAML: `initialize_fixed_points` logs `[IBMCloth] Warning: No fixed points defined` and continues (does not crash).
- Gravity + initial velocity produce ballistic motion before aerodynamic forces stabilize; body doesn't immediately NaN.

### 7.7 Stage 5 active control
- Scripted linear interpolation from cruise targets to landing targets over 500 frames produces a visible shape transition in the `.ply` animation.
- `cons_pos` mutations don't destabilize XPBD (patagium tracks the new targets within 2–3 substeps).

---

## Part 8 — Next steps, in order

The ladder treats "what to build" and "what to run" as interleaved. Concrete to-do:

**You (Nelly):**
1. Decide on the reference-.blend question: send me a hand-authored reference squirrel .blend (or just the OBJ + a screenshot) at one parameter set, OR tell me to work blind from anatomical references. Reply in chat; I'll plan accordingly.
2. Whenever Stage 0 is reached: run the sim, open `solid_0000.ply` in Blender, and tell me where the torso actually lands so we can tune `repose_position` and `fixed_region.min/max`.
3. Install Paraview (if not already) — 5.11 or later works fine.

**Me (Claude), next session:**
1. Extend `parametric_mesh/scripts/mesh_generator.py` per §1.4: real boundary curve → Grid Fill, remove cylinder body, fan out appendages, add camber/dihedral displacement hooks.
2. Add `patagium.camber` and `armature.bone_angles.wing_droop` to the config schema.
3. Generate the first real mesh, run the `meshio` validator, then hand off to you for Stage 0 run.
4. Later: add `set_control_targets` API to `IBMClothConfig` for Stage 5.

---

## Part 9 — Reference tables

### 9.1 Critical files

| File | Role | Status |
|------|------|--------|
| `parametric_mesh/scripts/mesh_generator.py` | Parametric Blender generator | Needs §1.4 rewrite |
| `parametric_mesh/config/armature/base_rig.yaml` | Bone lengths + angles | Needs `wing_droop` addition |
| `parametric_mesh/config/species/flying_squirrel_minimal.yaml` | Species patagium spline | Needs `camber:` block |
| `parametric_mesh/config/appendages/squirrel_standard.yaml` | Optional styliform processes | OK — generator just needs to read all 5 |
| `3D/assets/mesh/squirrel.obj` | Mesh → simulation | Generated by script |
| `3D/assets/mesh/squirrel_metadata.json` | Vertex groups, control indices | Generated by script |
| `3D/configs/squirrel.yaml` | Sim-side config (material, BCs) | Done |
| `3D/ibm_cloth_base.py` | Fixed-point selector | Done |
| `3D/init_conditions.py` | Uniform-inflow kernel | Done |
| `3D/hyperparameters.py` | Scenario + inflow_U knobs | Done |
| `3D/run.py` | Scenario dispatch at `init_vorts()` | Done |

### 9.2 YAML schema (squirrel.yaml) cheat sheet

| Key | Type | Notes |
|-----|------|-------|
| `metadata.name` | string | Resolves to `assets/mesh/<name>.obj` and `assets/mesh/<name>_metadata.json` |
| `material_properties.global_scale` | float | Post-normalization scale; 0.35 fits a squirrel in a 256×128×128 domain |
| `material_properties.repose_position` | [x, y, z] | Body center after scale, in domain coords [0,1] |
| `material_properties.regions.membrane.density` | float | `rho` for mass = area × rho |
| `material_properties.regions.membrane.length_constraint_alpha` | float | ↑ = more stretchy |
| `material_properties.regions.membrane.bend_constraint_alpha` | float | ↑ = floppier; ↓ = stiffer |
| `simulation.solve_iters` | int | XPBD iterations per step; 20 (shakedown) to 50 (production) |
| `simulation.dt` | float | Substep size; 0.0005 is the tested value |
| `simulation.gravity` | [x, y, z] | `[0,0,0]` wind-tunnel; `[0,0,-3]` free-glide |
| `constraints.fixed_region.type` | `bbox` or `indices` | See §3.2 |
| `constraints.fixed_region.min/max` | [x, y, z] | bbox only; domain coords |
| `constraints.fixed_region.values` | list[int] | indices only; or omit to read metadata.json |

### 9.3 Tuning knobs by layer

| Layer | Knob | Where | Effect |
|-------|------|-------|--------|
| Mesh resolution | `res_x, res_y, res_z` | `3D/hyperparameters.py:24-26` | Linear ↑ = memory + runtime |
| Sim duration | `total_frames` | `3D/hyperparameters.py:35` | Linear ↑ = runtime |
| Fluid stability | `CFL` | `3D/hyperparameters.py:32` | ↓ = smaller `dt`, more stable, slower |
| Viewer cadence | `visualize_dt` | `3D/hyperparameters.py:27` | ↓ = more frames written, more disk |
| GPU memory | `device_memory_GB` | `3D/run.py:12` | Match to stage; don't over-allocate |
| Solid stiffness | `length_constraint_alpha` | YAML | ↑ = stretchier |
| Solid bend | `bend_constraint_alpha` | YAML | ↑ = floppier (parachute-like); ↓ = rigid (airplane-like) |
| Inflow speed | `inflow_U` | `3D/hyperparameters.py` | Wind tunnel only; Reynolds number scales with this |
| XPBD iterations | `simulation.solve_iters` | YAML | ↑ = more stable at cost of runtime |

### 9.4 Doc map

| Document | What it covers |
|----------|----------------|
| **This file (FLYING_SQUIRREL_GUIDE.md)** | Complete guide + plan (canonical reference) |
| `CLAUDE.md` | Agent-facing repo orientation |
| `README.md` | Upstream SIGGRAPH paper acknowledgment (inherited) |
| `parametric_mesh/README.md` | Parametric subsystem overview |
| `~/.claude/plans/how-do-i-bring-whimsical-pancake.md` | Original implementation plan (superseded by this guide) |
