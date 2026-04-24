#!/usr/bin/env python3
"""
SVG Planiform Exporter
Generates top-down and side view SVG visualizations of gliding animal meshes

Usage:
    python planiform_exporter.py --blend-file path/to/animal.blend --output path/to/output.svg
"""

import sys
from pathlib import Path
import argparse

# Add parametric_mesh to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    import bpy
    import mathutils
    from mathutils import Vector
except ImportError:
    print("Error: This script must be run with Blender Python")
    print("Usage: blender --background --python planiform_exporter.py -- --blend-file file.blend")
    sys.exit(1)

import svgwrite
from svgwrite import mm


class PlaniformExporter:
    """Export planiform (top/side view) SVG from Blender mesh"""

    def __init__(self, output_path: str, view: str = 'top', show_bones: bool = True,
                 show_outline: bool = True, width: float = 200, height: float = 200):
        """
        Initialize exporter

        Args:
            output_path: Output SVG file path
            view: 'top', 'side', or 'front'
            show_bones: Include skeleton bones
            show_outline: Include mesh outline
            width/height: SVG dimensions in mm
        """
        self.output_path = Path(output_path)
        self.view = view
        self.show_bones = show_bones
        self.show_outline = show_outline

        # Create SVG drawing
        self.dwg = svgwrite.Drawing(str(self.output_path), size=(f'{width}mm', f'{height}mm'))
        self.dwg.viewbox(minx=0, miny=0, width=width, height=height)

        # Projection settings
        self.scale = 800  # Scale factor for Blender units to SVG units
        self.offset_x = width / 2
        self.offset_y = height / 2

    def project_point(self, point: Vector) -> tuple:
        """Project 3D point to 2D based on view"""
        if self.view == 'top':
            # Top view: XY plane (looking down Z axis)
            x, y = point.x, point.y
        elif self.view == 'side':
            # Side view: XZ plane (looking along Y axis)
            x, y = point.x, point.z
        elif self.view == 'front':
            # Front view: YZ plane (looking along X axis)
            x, y = point.y, point.z
        else:
            raise ValueError(f"Unknown view: {self.view}")

        # Scale and offset
        svg_x = x * self.scale + self.offset_x
        svg_y = -y * self.scale + self.offset_y  # Flip Y (SVG Y increases downward)

        return (svg_x, svg_y)

    def add_grid(self, grid_spacing: float = 0.1):
        """Add reference grid"""
        grid_group = self.dwg.g(id='grid', stroke='lightgray', stroke_width=0.5, fill='none')

        # Draw grid lines
        for i in range(-5, 6):
            pos = i * grid_spacing
            p1 = self.project_point(Vector((pos, -0.5, 0)))
            p2 = self.project_point(Vector((pos, 0.5, 0)))
            grid_group.add(self.dwg.line(start=p1, end=p2))

            p1 = self.project_point(Vector((-0.5, pos, 0)))
            p2 = self.project_point(Vector((0.5, pos, 0)))
            grid_group.add(self.dwg.line(start=p1, end=p2))

        self.dwg.add(grid_group)

    def add_bones(self, armature_obj):
        """Add bone skeleton to SVG"""
        if not armature_obj or armature_obj.type != 'ARMATURE':
            return

        bone_group = self.dwg.g(id='bones', stroke='black', stroke_width=2, fill='none')

        # Get armature in world space
        for bone in armature_obj.data.bones:
            # Get bone head and tail in world space
            head = armature_obj.matrix_world @ bone.head_local
            tail = armature_obj.matrix_world @ bone.tail_local

            # Project to 2D
            head_2d = self.project_point(head)
            tail_2d = self.project_point(tail)

            # Draw bone as line
            bone_group.add(self.dwg.line(start=head_2d, end=tail_2d))

            # Add small circle at joints
            bone_group.add(self.dwg.circle(center=head_2d, r=2, fill='black'))

        self.dwg.add(bone_group)

    def add_mesh_outline(self, mesh_obj):
        """Add mesh outline to SVG"""
        if not mesh_obj or mesh_obj.type != 'MESH':
            return

        outline_group = self.dwg.g(id='outline', stroke='darkblue', stroke_width=1.5, fill='lightblue', fill_opacity=0.3)

        # Get mesh vertices in world space
        mesh = mesh_obj.data
        world_verts = [mesh_obj.matrix_world @ v.co for v in mesh.vertices]

        # Project all vertices
        projected = [self.project_point(v) for v in world_verts]

        # Simple convex hull for outline (simplified - could use scipy)
        # For now, just draw all triangles
        for face in mesh.polygons:
            face_verts = [projected[v] for v in face.vertices]
            outline_group.add(self.dwg.polygon(points=face_verts))

        self.dwg.add(outline_group)

    def add_title(self, title: str):
        """Add title text"""
        title_group = self.dwg.g(id='title')
        title_group.add(self.dwg.text(
            title,
            insert=(10, 20),
            font_size='14px',
            font_family='Arial',
            fill='black'
        ))
        self.dwg.add(title_group)

    def export(self):
        """Save SVG file"""
        self.dwg.save()
        print(f"[Planiform] Exported {self.view} view to {self.output_path}")


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description='Export planiform SVG from Blender file')
    parser.add_argument('--blend-file', type=str, help='Input .blend file')
    parser.add_argument('--output', type=str, required=True, help='Output .svg file')
    parser.add_argument('--view', type=str, default='top', choices=['top', 'side', 'front'])
    parser.add_argument('--title', type=str, default='Gliding Animal Planiform')

    # Parse args (Blender passes args after --)
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)

    # Load blend file if specified
    if args.blend_file:
        bpy.ops.wm.open_mainfile(filepath=args.blend_file)
        print(f"[Planiform] Loaded {args.blend_file}")

    # Find armature and mesh objects
    armature_obj = None
    mesh_obj = None

    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and not armature_obj:
            armature_obj = obj
        elif obj.type == 'MESH' and not mesh_obj:
            mesh_obj = obj

    # Create exporter
    exporter = PlaniformExporter(
        output_path=args.output,
        view=args.view,
        show_bones=True,
        show_outline=True
    )

    # Add elements
    exporter.add_grid()
    exporter.add_title(args.title)

    if armature_obj:
        exporter.add_bones(armature_obj)
    if mesh_obj:
        exporter.add_mesh_outline(mesh_obj)

    # Export
    exporter.export()


if __name__ == "__main__":
    main()
