"""
Squirrel Stage-0 starter mesh generator.

Builds a flat top-view silhouette of a spread flying squirrel (head, torso strip,
propatagium, main patagium, uropatagium, tail) as a single triangulated surface
and writes it to 3D/assets/mesh/squirrel.obj for use with 3D/configs/squirrel.yaml.

Pure bpy + bmesh. No Hydra, no external deps. Runnable two ways:
    1. Blender Scripting tab → open this file, Alt+P to run.
    2. Headless: blender --background --python parametric_mesh/scripts/starter_squirrel.py

See FLYING_SQUIRREL_GUIDE.md §1.2 (target silhouette) and §1.3 (mesh validator).
This is a throwaway starter; the long-term home is parametric_mesh/scripts/mesh_generator.py.
"""

import math
import sys
from pathlib import Path

import bpy
import bmesh


# --- Knobs ---------------------------------------------------------------
# Units are arbitrary. TrianMesh normalizes to unit bbox before the sim runs,
# so only proportions matter here.

SPAN            = 2.0    # wing-tip to wing-tip (Y axis)
BODY_LENGTH     = 1.6    # head to tail tip (X axis)
NECK_WIDTH      = 0.25   # body width at neck
TAIL_WIDTH      = 0.18   # body width at tail base

HEAD_FRAC       = 0.05   # head tip X position (frac of BODY_LENGTH)
WRIST_FRAC      = 0.28   # forelimb tip X
ANKLE_FRAC      = 0.62   # hindlimb tip X
TAIL_BASE_FRAC  = 0.82   # where body narrows back toward the tail
TAIL_TIP_FRAC   = 0.98   # tail tip X

PATAGIUM_BULGE  = 0.06   # outward curve of trailing edge between wrist↔ankle (frac of SPAN)
LEADING_SAMPLES = 4      # interior samples on neck→wrist edge (propatagium curvature)
TRAILING_SAMPLES = 8     # interior samples on wrist→ankle edge (main patagium trailing)

# Density: subdivide_edges cuts applied after triangle_fill.
#   0 → coarse  (~40 tris)        — useful for sanity-checking topology
#   1 → ~200 tris
#   2 → ~800 tris                 — Stage 0 shakedown
#   3 → ~3k tris                  — Stage 1 material tune
#   4 → ~12k tris                 — Stage 2 production
RESOLUTION_CUTS = 2

OUTPUT_NAME = "squirrel"

# Repo root. Auto-resolves from __file__ when available (headless run, or
# Text Editor block loaded from disk). Falls back to the hard-coded path
# if Blender is running this as an unsaved text block — set it here if you
# move the repo.
REPO_ROOT_FALLBACK = Path("/home/nelly/repo/nranthony/pfm-sfi")


def _resolve_repo_root() -> Path:
    try:
        here = Path(__file__).resolve()
    except NameError:
        return REPO_ROOT_FALLBACK
    # Expected layout: <repo>/parametric_mesh/scripts/starter_squirrel.py
    if len(here.parents) >= 3 and (here.parents[2] / "3D").is_dir():
        return here.parents[2]
    return REPO_ROOT_FALLBACK


OUTPUT_DIR = _resolve_repo_root() / "3D" / "assets" / "mesh"


# --- Silhouette ----------------------------------------------------------

def build_boundary_points():
    """Boundary vertices tracing the silhouette perimeter counter-clockwise.
    +X = head→tail, +Y = left span, -Y = right span, Z = 0 (flat rest pose)."""
    s = SPAN / 2.0
    bl = BODY_LENGTH

    head_tip  = (HEAD_FRAC * bl, 0.0, 0.0)
    neck_x    = (HEAD_FRAC + 0.05) * bl
    wrist_x   = WRIST_FRAC * bl
    ankle_x   = ANKLE_FRAC * bl
    tail_base = (TAIL_BASE_FRAC * bl, -TAIL_WIDTH / 2.0, 0.0)
    tail_tip  = (TAIL_TIP_FRAC * bl, 0.0, 0.0)

    pts = [head_tip]

    # head → neck taper (right side)
    pts.append((neck_x, -NECK_WIDTH / 2.0, 0.0))

    # neck → wrist (propatagium leading edge, curves outward)
    for i in range(1, LEADING_SAMPLES + 1):
        t = i / (LEADING_SAMPLES + 1)
        x = neck_x * (1 - t) + wrist_x * t
        y_inner = -NECK_WIDTH / 2.0
        y_outer = -s
        # ease-out so curve tightens near wrist
        pts.append((x, y_inner * (1 - t) + y_outer * (t ** 0.7), 0.0))
    pts.append((wrist_x, -s, 0.0))

    # wrist → ankle (main patagium trailing edge, sinusoidal bulge)
    for i in range(1, TRAILING_SAMPLES + 1):
        t = i / (TRAILING_SAMPLES + 1)
        x = wrist_x * (1 - t) + ankle_x * t
        bulge = PATAGIUM_BULGE * SPAN * math.sin(math.pi * t)
        pts.append((x, -(s + bulge), 0.0))
    pts.append((ankle_x, -s, 0.0))

    # ankle → tail base (uropatagium, straight taper)
    pts.append(tail_base)

    # tail base → tail tip
    pts.append(tail_tip)

    # mirror the right-side perimeter to produce the left side
    right_side = pts[1:-1]
    for (x, y, z) in reversed(right_side):
        pts.append((x, -y, z))

    return pts


# --- Mesh construction ---------------------------------------------------

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.curves):
        bpy.data.curves.remove(block)


def build_mesh(name, points, cuts):
    bm = bmesh.new()

    verts = [bm.verts.new(p) for p in points]
    bm.verts.ensure_lookup_table()

    # Close the boundary loop.
    boundary_edges = [
        bm.edges.new([verts[i], verts[(i + 1) % len(verts)]])
        for i in range(len(verts))
    ]

    # Fill interior with triangles (ear-clipping; robust for star-shaped polygons).
    bmesh.ops.triangle_fill(bm, edges=boundary_edges, use_beauty=True)

    # Subdivide interior for resolution.
    if cuts > 0:
        bmesh.ops.subdivide_edges(
            bm,
            edges=bm.edges[:],
            cuts=cuts,
            use_grid_fill=False,
        )

    # Convert any quads from subdivide back to tris.
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    # Outward normals (not load-bearing for IBM coupling but keeps Paraview shading sane).
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()

    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    return obj


def export_obj(obj, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # forward_axis/up_axis declare the content frame.
    # Silhouette is built with +X = head→tail, +Z = up; match that here so the
    # OBJ preserves those axes verbatim and TrianMesh loads it unchanged.
    bpy.ops.wm.obj_export(
        filepath=str(out_path),
        export_selected_objects=True,
        apply_modifiers=True,
        export_triangulated_mesh=True,
        export_materials=False,
        forward_axis='X',
        up_axis='Z',
    )


# --- Validator (mirrors FSG §1.3) ----------------------------------------

def validate(obj):
    import numpy as np
    v = np.array([list(vt.co) for vt in obj.data.vertices])
    f = np.array([[p.vertices[0], p.vertices[1], p.vertices[2]] for p in obj.data.polygons])
    edges = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int((counts == 1).sum())
    interior = int((counts == 2).sum())
    broken = int((counts > 2).sum())
    print(f"[starter_squirrel] validator: verts={len(v)} tris={len(f)} "
          f"boundary_edges={boundary} interior_edges={interior} broken_edges={broken}")
    if broken > 0:
        print("[starter_squirrel] WARNING: broken_edges > 0 — mesh has non-manifold interior.")
    if boundary == 0:
        print("[starter_squirrel] WARNING: boundary_edges = 0 — expected an open silhouette.")


# --- Entry point ---------------------------------------------------------

def main():
    print(f"[starter_squirrel] Blender {bpy.app.version_string}")
    clear_scene()

    points = build_boundary_points()
    print(f"[starter_squirrel] boundary verts: {len(points)}")

    obj = build_mesh(OUTPUT_NAME, points, cuts=RESOLUTION_CUTS)
    print(f"[starter_squirrel] mesh: {len(obj.data.vertices)} verts, {len(obj.data.polygons)} tris")

    out_path = OUTPUT_DIR / f"{OUTPUT_NAME}.obj"
    export_obj(obj, out_path)
    print(f"[starter_squirrel] wrote: {out_path}")

    validate(obj)


if __name__ == "__main__":
    main()
