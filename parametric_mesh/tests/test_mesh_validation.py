#!/usr/bin/env python3
"""
Test Mesh Validation
Validates generated mesh files meet requirements
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    print("Warning: meshio not available, skipping mesh tests")


class TestMeshValidation(unittest.TestCase):
    """Test mesh validation against FLYING_SQUIRREL_GUIDE.md §1.3"""

    def setUp(self):
        self.test_outputs = Path(__file__).parent / "outputs"
        self.test_outputs.mkdir(exist_ok=True)

    @unittest.skipIf(not MESHIO_AVAILABLE, "meshio not installed")
    def test_mesh_is_triangulated(self):
        """Verify mesh contains only triangular faces"""
        # This will test generated meshes
        mesh_dir = Path(__file__).parent.parent / "generated" / "meshes"

        if not mesh_dir.exists():
            self.skipTest("No generated meshes found")

        obj_files = list(mesh_dir.glob("*.obj"))
        if not obj_files:
            self.skipTest("No .obj files in generated/meshes")

        for obj_file in obj_files:
            print(f"\nValidating: {obj_file.name}")
            mesh = meshio.read(obj_file)

            # Check for triangle cells
            self.assertIn('triangle', mesh.cells_dict,
                          f"{obj_file.name}: No triangle cells found")

            # Check no other face types
            for cell_type in mesh.cells_dict.keys():
                if cell_type not in ['triangle', 'vertex', 'line']:
                    self.fail(f"{obj_file.name}: Contains {cell_type} (only triangles allowed)")

            print(f"  ✓ All faces are triangles ({len(mesh.cells_dict['triangle'])} tris)")

    @unittest.skipIf(not MESHIO_AVAILABLE, "meshio not installed")
    def test_mesh_vertex_count(self):
        """Verify mesh has reasonable vertex count"""
        mesh_dir = Path(__file__).parent.parent / "generated" / "meshes"

        if not mesh_dir.exists():
            self.skipTest("No generated meshes found")

        obj_files = list(mesh_dir.glob("*.obj"))
        if not obj_files:
            self.skipTest("No .obj files in generated/meshes")

        for obj_file in obj_files:
            mesh = meshio.read(obj_file)

            vertex_count = mesh.points.shape[0]
            triangle_count = len(mesh.cells_dict.get('triangle', []))

            # Check vertex count is in reasonable range
            self.assertGreater(vertex_count, 10,
                               f"{obj_file.name}: Too few vertices ({vertex_count})")
            self.assertLess(vertex_count, 100000,
                            f"{obj_file.name}: Too many vertices ({vertex_count})")

            # Check triangle count
            self.assertGreater(triangle_count, 0,
                               f"{obj_file.name}: No triangles found")

            print(f"  ✓ Vertex count: {vertex_count}")
            print(f"  ✓ Triangle count: {triangle_count}")

    @unittest.skipIf(not MESHIO_AVAILABLE, "meshio not installed")
    def test_mesh_has_no_degenerate_triangles(self):
        """Verify no zero-area triangles"""
        mesh_dir = Path(__file__).parent.parent / "generated" / "meshes"

        if not mesh_dir.exists():
            self.skipTest("No generated meshes found")

        obj_files = list(mesh_dir.glob("*.obj"))
        if not obj_files:
            self.skipTest("No .obj files in generated/meshes")

        for obj_file in obj_files:
            mesh = meshio.read(obj_file)

            if 'triangle' not in mesh.cells_dict:
                continue

            vertices = mesh.points
            triangles = mesh.cells_dict['triangle']

            degenerate_count = 0
            for i, tri in enumerate(triangles):
                v0, v1, v2 = vertices[tri]

                # Compute triangle area using cross product
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = np.linalg.norm(cross) / 2.0

                if area < 1e-10:
                    degenerate_count += 1

            self.assertEqual(degenerate_count, 0,
                             f"{obj_file.name}: Found {degenerate_count} degenerate triangles")

            print(f"  ✓ No degenerate triangles")

    @unittest.skipIf(not MESHIO_AVAILABLE, "meshio not installed")
    def test_mesh_bounds(self):
        """Verify mesh is within reasonable bounds"""
        mesh_dir = Path(__file__).parent.parent / "generated" / "meshes"

        if not mesh_dir.exists():
            self.skipTest("No generated meshes found")

        obj_files = list(mesh_dir.glob("*.obj"))
        if not obj_files:
            self.skipTest("No .obj files in generated/meshes")

        for obj_file in obj_files:
            mesh = meshio.read(obj_file)
            vertices = mesh.points

            # Compute bounds
            min_bounds = vertices.min(axis=0)
            max_bounds = vertices.max(axis=0)
            extents = max_bounds - min_bounds

            print(f"  Bounds X: [{min_bounds[0]:.3f}, {max_bounds[0]:.3f}]")
            print(f"  Bounds Y: [{min_bounds[1]:.3f}, {max_bounds[1]:.3f}]")
            print(f"  Bounds Z: [{min_bounds[2]:.3f}, {max_bounds[2]:.3f}]")
            print(f"  Extents: {extents}")

            # Check extents are reasonable (not too small or huge)
            for i, axis in enumerate(['X', 'Y', 'Z']):
                self.assertGreater(extents[i], 0.0,
                                   f"{obj_file.name}: {axis} extent is zero")
                self.assertLess(extents[i], 10.0,
                                f"{obj_file.name}: {axis} extent {extents[i]} too large")

            print(f"  ✓ Bounds are reasonable")

    def test_metadata_json_exists(self):
        """Verify metadata JSON is generated"""
        mesh_dir = Path(__file__).parent.parent / "generated" / "meshes"

        if not mesh_dir.exists():
            self.skipTest("No generated meshes found")

        json_files = list(mesh_dir.glob("*_metadata.json"))

        if not json_files:
            self.skipTest("No metadata JSON files found")

        for json_file in json_files:
            import json
            with open(json_file) as f:
                metadata = json.load(f)

            # Check required fields
            self.assertIn('name', metadata)
            self.assertIn('version', metadata)
            self.assertIn('trunk_length', metadata)

            print(f"  ✓ Metadata valid: {json_file.name}")
            print(f"    Name: {metadata['name']}")
            print(f"    Version: {metadata['version']}")


class TestBlenderOutputs(unittest.TestCase):
    """Test Blender-generated outputs"""

    def setUp(self):
        self.generated_dir = Path(__file__).parent.parent / "generated"

    def test_blend_file_exists(self):
        """Verify .blend files are generated"""
        mesh_dir = self.generated_dir / "meshes"

        if not mesh_dir.exists():
            self.skipTest("No generated meshes directory")

        blend_files = list(mesh_dir.glob("*.blend"))

        if not blend_files:
            print("  ⚠ No .blend files found (may not have run generator yet)")
            self.skipTest("No .blend files generated yet")

        for blend_file in blend_files:
            # Check file size is reasonable
            file_size = blend_file.stat().st_size
            self.assertGreater(file_size, 1000,
                               f"{blend_file.name}: File too small ({file_size} bytes)")

            print(f"  ✓ {blend_file.name} exists ({file_size / 1024:.1f} KB)")

    def test_obj_file_exists(self):
        """Verify .obj files are generated"""
        mesh_dir = self.generated_dir / "meshes"

        if not mesh_dir.exists():
            self.skipTest("No generated meshes directory")

        obj_files = list(mesh_dir.glob("*.obj"))

        if not obj_files:
            print("  ⚠ No .obj files found (may not have run generator yet)")
            self.skipTest("No .obj files generated yet")

        for obj_file in obj_files:
            # Check file is not empty
            file_size = obj_file.stat().st_size
            self.assertGreater(file_size, 100,
                               f"{obj_file.name}: File too small or empty")

            print(f"  ✓ {obj_file.name} exists ({file_size / 1024:.1f} KB)")

    def test_svg_planiform_exists(self):
        """Verify SVG planiform files can be generated"""
        planform_dir = self.generated_dir / "planforms"

        if not planform_dir.exists():
            print("  ⚠ No planforms directory (SVG export not run yet)")
            self.skipTest("No planforms directory")

        svg_files = list(planform_dir.glob("*.svg"))

        if not svg_files:
            print("  ⚠ No SVG files (planiform export not run yet)")
            self.skipTest("No SVG files generated yet")

        for svg_file in svg_files:
            # Check file contains SVG content
            content = svg_file.read_text()
            self.assertIn('<svg', content, f"{svg_file.name}: Not a valid SVG file")
            self.assertIn('</svg>', content, f"{svg_file.name}: SVG not closed")

            print(f"  ✓ {svg_file.name} is valid SVG")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Mesh Validation")
    print("=" * 70)
    unittest.main(verbosity=2)
