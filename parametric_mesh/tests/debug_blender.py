#!/usr/bin/env python3
"""
Debug Blender Mesh Generation
Run inside Blender to debug mesh generation issues

Usage:
    blender --background --python debug_blender.py
"""

import sys
from pathlib import Path

# Check if running in Blender
try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("❌ This script must be run inside Blender")
    print("Usage: blender --background --python debug_blender.py")
    sys.exit(1)

import mathutils
from mathutils import Vector


def check_blender_environment():
    """Check Blender environment"""
    print("\n" + "="*70)
    print("BLENDER ENVIRONMENT")
    print("="*70)

    print(f"✓ Blender version: {bpy.app.version_string}")
    print(f"✓ Blender build: {bpy.app.build_date.decode()}")
    print(f"✓ Python version: {sys.version}")

    # Check bpy modules
    required_modules = ['ops', 'data', 'context', 'types']
    for mod in required_modules:
        if hasattr(bpy, mod):
            print(f"✓ bpy.{mod} available")
        else:
            print(f"❌ bpy.{mod} NOT available")


def test_basic_operations():
    """Test basic Blender operations"""
    print("\n" + "="*70)
    print("TESTING BASIC OPERATIONS")
    print("="*70)

    # Clear scene
    print("\nClearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    print("✓ Scene cleared")

    # Create cube
    print("\nCreating test cube...")
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    cube = bpy.context.active_object
    print(f"✓ Cube created: {cube.name}")
    print(f"  Vertices: {len(cube.data.vertices)}")
    print(f"  Faces: {len(cube.data.polygons)}")

    # Create armature
    print("\nCreating test armature...")
    armature_data = bpy.data.armatures.new('TestArmature')
    armature_obj = bpy.data.objects.new('TestArmature', armature_data)
    bpy.context.collection.objects.link(armature_obj)
    print(f"✓ Armature created: {armature_obj.name}")

    # Enter edit mode
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Add bone
    bone = armature_data.edit_bones.new('TestBone')
    bone.head = Vector((0, 0, 0))
    bone.tail = Vector((0, 0, 1))

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"✓ Bone added: {bone.name}")

    print("\n✓ Basic operations work")


def test_armature_generation():
    """Test armature generation similar to mesh_generator.py"""
    print("\n" + "="*70)
    print("TESTING ARMATURE GENERATION")
    print("="*70)

    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Create armature
    armature_data = bpy.data.armatures.new('DebugArmature')
    armature_obj = bpy.data.objects.new('DebugArmature', armature_data)
    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj

    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature_data.edit_bones

    # Create spine chain
    print("\nCreating spine chain...")
    trunk_len = 0.20
    spine_names = ['Spine_Pelvis', 'Spine_Lumbar', 'Spine_Thoracic']
    spine_lengths = [0.15, 0.20, 0.25]

    pos = Vector((0, 0, 0))
    spine_bones = []

    for i, (name, length) in enumerate(zip(spine_names, spine_lengths)):
        bone = edit_bones.new(name)
        bone.head = pos
        bone.tail = pos + Vector((length * trunk_len, 0, 0))

        if i > 0:
            bone.parent = spine_bones[-1]

        spine_bones.append(bone)
        pos = bone.tail
        print(f"  ✓ Created {name}: head={bone.head}, tail={bone.tail}")

    # Create limb
    print("\nCreating test limb...")
    limb_bone = edit_bones.new('Limb_Test')
    limb_bone.head = spine_bones[-1].head
    limb_bone.tail = limb_bone.head + Vector((0, 0.15 * trunk_len, 0))
    limb_bone.parent = spine_bones[-1]
    print(f"  ✓ Created {limb_bone.name}")

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"\n✓ Armature created with {len(armature_data.bones)} bones:")
    for bone in armature_data.bones:
        print(f"  - {bone.name}")


def test_mesh_creation():
    """Test basic mesh creation"""
    print("\n" + "="*70)
    print("TESTING MESH CREATION")
    print("="*70)

    # Create simple mesh
    print("\nCreating simple mesh...")

    mesh_data = bpy.data.meshes.new('TestMesh')
    verts = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0)
    ]
    faces = [(0, 1, 2, 3)]

    mesh_data.from_pydata(verts, [], faces)
    mesh_obj = bpy.data.objects.new('TestMesh', mesh_data)
    bpy.context.collection.objects.link(mesh_obj)

    print(f"✓ Mesh created: {mesh_obj.name}")
    print(f"  Vertices: {len(mesh_obj.data.vertices)}")
    print(f"  Faces: {len(mesh_obj.data.polygons)}")

    # Triangulate
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"✓ Triangulated: {len(mesh_obj.data.polygons)} triangles")


def test_file_export():
    """Test file export"""
    print("\n" + "="*70)
    print("TESTING FILE EXPORT")
    print("="*70)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Export blend file
    blend_path = output_dir / "debug_test.blend"
    print(f"\nExporting .blend: {blend_path}")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if blend_path.exists():
        print(f"✓ .blend exported ({blend_path.stat().st_size} bytes)")
    else:
        print("❌ .blend export failed")

    # Export OBJ
    obj_path = output_dir / "debug_test.obj"
    print(f"\nExporting .obj: {obj_path}")

    # Select all mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)

    if bpy.context.selected_objects:
        bpy.ops.wm.obj_export(
            filepath=str(obj_path),
            export_selected_objects=True,
            apply_modifiers=True,
            export_triangulated_mesh=True
        )
        if obj_path.exists():
            print(f"✓ .obj exported ({obj_path.stat().st_size} bytes)")
        else:
            print("❌ .obj export failed")
    else:
        print("⚠ No mesh objects to export")


def main():
    """Run all debug tests"""
    print("\n" + "="*70)
    print("BLENDER MESH GENERATION DEBUG")
    print("="*70)

    tests = [
        ("Environment Check", check_blender_environment),
        ("Basic Operations", test_basic_operations),
        ("Armature Generation", test_armature_generation),
        ("Mesh Creation", test_mesh_creation),
        ("File Export", test_file_export),
    ]

    results = []

    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except Exception as e:
            print(f"\n❌ {name} FAILED:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "="*70)
    print("DEBUG SUMMARY")
    print("="*70)

    for name, passed, error in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")

    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)

    print(f"\nResults: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✓ All debug tests passed!")
        return 0
    else:
        print("\n❌ Some debug tests failed")
        return 1


if __name__ == '__main__':
    if IN_BLENDER:
        sys.exit(main())
