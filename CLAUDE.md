# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SIGGRAPH Asia 2024 research code for solid-fluid interaction on particle flow maps (PFM-SFI). Fork/extension focused on adapting the 3D IBM-cloth solver to simulate **deformable mammalian gliders** (flying squirrel, sugar glider, colugo) using parametrically generated meshes.

Two simulators live side-by-side and do not share code:
- `2D/` — swimming fish example, MPM + multigrid PCG, ~3 GB GPU
- `3D/` — cloth/flag/parachute/glider, IBM + XPBD, ~10.5 GB GPU

Both require an NVIDIA GPU with CUDA. Dependencies are managed via `environment.yml` (conda/mamba, recommended — env name is `flowmap`) or `requirements.txt` (pip — user must install CUDA toolkit separately).

## Commands

```bash
# Environment
conda env create -f environment.yml && conda activate flowmap   # conda name is "flowmap", not "pfm-sfi"
python -c "import taichi as ti; ti.init(arch=ti.cuda)"          # verify CUDA

# Run simulations (must cd into the directory — both use relative imports)
cd 2D && python run.py
cd 3D && python run.py

# Run 3D with a custom parametric mesh (YAML mode, see "Custom mesh pipeline" below)
cd 3D && IBM_CLOTH_MODE=yaml IBM_CLOTH_CONFIG=configs/<name>.yaml python run.py

# Parametric mesh generation (requires Blender 3.6+ / 4.x on PATH or set BLENDER_PATH)
cd parametric_mesh
pip install -r requirements.txt                                  # hydra-core, omegaconf, svgwrite, pyyaml
blender --background --python scripts/mesh_generator.py          # writes to generated/meshes/
blender --background --python scripts/planiform_exporter.py -- \
    --blend-file generated/meshes/<name>.blend \
    --output generated/planforms/<name>_top.svg --view top

# Cross-platform Blender runner (auto-resolves from BLENDER_PATH env / config/blender.yaml / PATH)
python scripts/blender_wrapper.py --script mesh_generator.py
# Windows convenience wrappers:
generate_mesh.bat
generate_planiform.bat

# Parametric mesh tests (no Blender needed for the first two)
cd parametric_mesh
python tests/test_config_loading.py
python tests/test_simulation_integration.py
python tests/debug_config.py --all
./tests/test_end_to_end.sh                                       # full pipeline incl. Blender
```

There is no lint/format configuration in the repo; do not introduce one unless asked.

## Simulator architecture (3D)

`3D/run.py` is a monolithic script. Control flow is driven by top-level imports that execute immediately:

1. `hyperparameters.py` — all sim constants (grid `res_x/y/z = 256/128/128`, `total_frames`, `CFL`, `exp_name`, encoder/neural-buffer sizes). Changing the simulation means editing this file — values are `from ... import *`-ed everywhere.
2. `ti.init(arch=ti.cuda, device_memory_GB=10.5)` is called in `run.py:12` before any Taichi fields are allocated.
3. `from ibm_cloth import *` (run.py:15) constructs the solid — this is where custom-mesh integration happens.
4. `init_conditions.py` seeds the fluid; `mgpcg.py` is the pressure solver; `io_utils.py` handles frame dumps under `logs/<exp_name>/`.

The solid side (IBM + XPBD) is factored into:
- `gmesh.TrianMesh` — loads triangulated vertices/faces, auto-normalizes to unit cube then applies `scale` and `repose` translation. Computes per-vertex inverse mass from triangle areas × `rho`.
- `framework.pbd_framework` — XPBD integrator holding position/velocity fields and a list of constraints.
- `length.LengthCons` + `bend.Bend3D` — edge-length and dihedral-bend constraints (alphas control stiffness).
- `ibm_cloth.py` — glue. Defines the IBM kernel (`ibm_kernel`, `sample_ibm_u`, `spread_force`, `update_force`, `advect_ibm`, `solve_for_xpbd`) that transfers velocity from the Eulerian grid to the mesh and spreads XPBD forces back.

Fixed-point (Dirichlet) boundary vertices are selected in `ibm_cloth_base.IBMClothConfig.initialize_fixed_points` with three-level precedence: (1) `constraints.fixed_region` in YAML — `type: bbox` with `min`/`max` in post-normalization domain coords, or `type: indices` reading `metadata['fixed_indices']`; (2) `fixed_indices` array in `<mesh>_metadata.json`; (3) legacy spatial rule `x < 0.205` (silk-flag fallback).

## Custom mesh pipeline

This is the main active work stream. Flow is **parametric config → Blender → OBJ + metadata → 3D simulator**:

```
parametric_mesh/config/*.yaml  (Hydra: armature + species + material + appendages + blender)
        │   hydra.main composes these; override via CLI (e.g. armature.trunk_length=0.25)
        ▼
parametric_mesh/scripts/mesh_generator.py   (runs INSIDE Blender)
  ├─ clear scene, build armature from bone_lengths × trunk_length
  ├─ apply bone spreads (forelimb/hindlimb) as rest pose
  ├─ add appendages (styloid etc.) if enabled
  ├─ generate body mesh (currently a placeholder cylinder)
  ├─ generate patagium from bezier splines between named anchors (wrist↔ankle, neck↔wrist, …)
  ├─ triangulate, assign vertex groups
  └─ export generated/meshes/<name>.{blend,obj} + <name>_metadata.json
        │
        ▼
3D/assets/mesh/<name>.obj          ← copy (or symlink) OBJ here
3D/assets/mesh/<name>_metadata.json ← ibm_cloth_base looks it up next to the OBJ
3D/configs/<name>.yaml              ← sim-side config (material + simulation blocks)
        │
        ▼
3D/ibm_cloth.py   (MODE='yaml' via IBM_CLOTH_MODE env var)
  └─ IBMClothConfig(config_path=...).setup()
        ├─ obj_parser → meshio.read → TrianMesh(verts, faces, rho, scale, repose)
        ├─ LengthCons + Bend3D with alphas from material_properties.regions.membrane
        └─ initialize_fixed_points (three-level precedence, see below)
        │
        ▼
3D/run.py picks up `mesh`, `xpbd`, `solve_iters`, `dt`, `g`, force fields, and
`cons_*` from the `ibm_cloth` module namespace via `from ibm_cloth import *`.
```

Key integration seams to know about:

- **Mode switching** (`3D/ibm_cloth.py:30`): `IBM_CLOTH_MODE=yaml|legacy` + `IBM_CLOTH_CONFIG=<path>`. Legacy mode hardcodes `silk2.obj` with flag parameters — preserving this is a stated backward-compatibility constraint, don't break it.
- **Config split**: the **parametric_mesh** Hydra configs drive *geometry generation*; the **3D/configs** YAML drives *simulation setup*. They currently share schema (metadata/material_properties/simulation blocks) — `ibm_cloth_base.IBMClothConfig.load_config` reads the sim-side YAML and locates the `.obj` via `config.metadata.name + '.obj'` under `3D/assets/mesh/`.
- **Metadata JSON** (written by mesh_generator, read by ibm_cloth_base) is the designated channel for passing vertex groups, anchor points, and material regions from Blender into the solver. It is loaded but not yet used for fixed-point selection — that wiring is unfinished.
- **Mesh requirements** are strict: triangles only (no quads/n-gons), outward normals, ~5k–15k tris. `meshio` loads via `cells_dict['triangle']` and will `KeyError` on quads. Single-layer silhouettes are fine — do NOT need to be watertight (the silk flag example is open). Full discussion in `FLYING_SQUIRREL_GUIDE.md` §1.
- **Normalization**: `TrianMesh` recenters by centroid and rescales to a unit bounding box before applying `scale` and `repose`. Absolute Blender units don't matter; proportions and orientation do (+X forward, +Z up).

Gotcha: `3D/ibm_cloth_base.py:81` builds the mesh path with `os.getcwd()`, so 3D scripts only work when run with `cwd = 3D/`.

## Design docs (read before large changes)

- `FLYING_SQUIRREL_GUIDE.md` — canonical project reference: mesh design, curvature strategy, YAML schema, staged iteration ladder, verification steps. SSoT for everything the glider work touches.

## State of the code

- `3D/ibm_backup/ibm_cloth_original.py` is the pre-refactor hardcoded version kept for reference — do not edit.
- `parametric_mesh/generated/` is output-only and starts empty; `logs/` is gitignored.
- Phase 2+ of the mesh pipeline (real boundary-fill patagium, bone-driven camber/dihedral displacement, active-control API on `IBMClothConfig`) is not implemented yet — see `FLYING_SQUIRREL_GUIDE.md` §1.4 and §6.7.
