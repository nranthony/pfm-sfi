#!/usr/bin/env python3
"""
Parametric Mesh Generator for Gliding Animals
Generates armature, patagium, and body mesh from Hydra config

Usage:
    blender --background --python mesh_generator.py -- --config-name=flying_squirrel_minimal
"""

import sys
import os
from pathlib import Path

# Add parametric_mesh to path for imports
script_dir = Path(__file__).parent
repo_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(repo_root / "parametric_mesh"))

import bpy
import mathutils
from mathutils import Vector
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml


class ParametricMeshGenerator:
    """Main class for generating parametric gliding animal meshes"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.armature_obj = None
        self.armature = None
        self.mesh_obj = None
        self.anchor_points = {}  # Store world positions of anchor points

    def generate(self):
        """Main generation pipeline"""
        print(f"[Generator] Starting mesh generation: {self.cfg.metadata.name}")

        # Clear existing scene
        self.clear_scene()

        # Phase 1: Create armature
        print("[Generator] Phase 1: Creating armature...")
        self.create_armature()
        self.apply_bone_transforms()

        # Phase 2: Add optional appendages
        print("[Generator] Phase 2: Adding appendages...")
        self.add_appendages()

        # Phase 3: Generate body mesh
        print("[Generator] Phase 3: Generating body mesh...")
        self.generate_body_mesh()

        # Phase 4: Generate patagium
        print("[Generator] Phase 4: Generating patagium...")
        self.generate_patagium()

        # Phase 5: Assign material regions (vertex groups)
        print("[Generator] Phase 5: Assigning material regions...")
        self.assign_material_regions()

        # Phase 6: Validate and triangulate
        print("[Generator] Phase 6: Validating and triangulating...")
        self.triangulate_and_validate()

        # Phase 7: Export files
        print("[Generator] Phase 7: Exporting files...")
        self.export_files()

        print(f"[Generator] Generation complete: {self.cfg.metadata.name}")

    def clear_scene(self):
        """Remove all objects from scene"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def create_armature(self):
        """Create base armature skeleton"""
        # Create armature object
        armature_data = bpy.data.armatures.new('Armature')
        self.armature_obj = bpy.data.objects.new('Armature', armature_data)
        bpy.context.collection.objects.link(self.armature_obj)
        bpy.context.view_layer.objects.active = self.armature_obj

        # Enter edit mode to add bones
        bpy.ops.object.mode_set(mode='EDIT')
        edit_bones = armature_data.edit_bones

        trunk_len = self.cfg.armature.trunk_length

        # Create spine chain
        spine_bones = []
        spine_names = ['Spine_Pelvis', 'Spine_Lumbar', 'Spine_Thoracic_Lower',
                       'Spine_Thoracic_Upper', 'Neck']
        spine_lengths = [
            self.cfg.armature.bone_lengths.spine.pelvis,
            self.cfg.armature.bone_lengths.spine.lumbar,
            self.cfg.armature.bone_lengths.spine.thoracic_lower,
            self.cfg.armature.bone_lengths.spine.thoracic_upper,
            self.cfg.armature.bone_lengths.spine.neck
        ]

        pos = Vector((0, 0, 0))
        for i, (name, length) in enumerate(zip(spine_names, spine_lengths)):
            bone = edit_bones.new(name)
            bone.head = pos
            bone.tail = pos + Vector((length * trunk_len, 0, 0))
            if i > 0:
                bone.parent = spine_bones[-1]
            spine_bones.append(bone)
            pos = bone.tail

        # Create head
        head_bone = edit_bones.new('Head')
        head_bone.head = spine_bones[-1].tail
        head_bone.tail = head_bone.head + Vector((0.08 * trunk_len, 0, 0))
        head_bone.parent = spine_bones[-1]

        # Create tail chain
        tail_bones = []
        tail_names = ['Tail_Base', 'Tail_Mid', 'Tail_Tip']
        tail_lengths = [
            self.cfg.armature.bone_lengths.tail.base,
            self.cfg.armature.bone_lengths.tail.mid,
            self.cfg.armature.bone_lengths.tail.tip
        ]

        pos = spine_bones[0].head  # Start from pelvis
        for i, (name, length) in enumerate(zip(tail_names, tail_lengths)):
            bone = edit_bones.new(name)
            bone.head = pos
            bone.tail = pos + Vector((-length * trunk_len, 0, 0))
            if i > 0:
                bone.parent = tail_bones[-1]
            else:
                bone.parent = spine_bones[0]
            tail_bones.append(bone)
            pos = bone.tail

        # Create limbs (left side, will mirror later)
        # Forelimb
        shoulder_bone = edit_bones.new('Shoulder_L')
        shoulder_bone.head = spine_bones[3].head  # Upper thoracic
        shoulder_bone.tail = shoulder_bone.head + Vector((0, 0.05 * trunk_len, 0))

        upper_arm_bone = edit_bones.new('Upper_Arm_L')
        upper_arm_bone.head = shoulder_bone.tail
        upper_arm_bone.tail = upper_arm_bone.head + Vector((0, self.cfg.armature.bone_lengths.forelimb.shoulder_to_elbow * trunk_len, 0))
        upper_arm_bone.parent = shoulder_bone

        forearm_bone = edit_bones.new('Forearm_L')
        forearm_bone.head = upper_arm_bone.tail
        forearm_bone.tail = forearm_bone.head + Vector((0, self.cfg.armature.bone_lengths.forelimb.elbow_to_wrist * trunk_len, 0))
        forearm_bone.parent = upper_arm_bone

        wrist_bone = edit_bones.new('Wrist_L')
        wrist_bone.head = forearm_bone.tail
        wrist_bone.tail = wrist_bone.head + Vector((0, self.cfg.armature.bone_lengths.forelimb.wrist_to_fingertip * trunk_len, 0))
        wrist_bone.parent = forearm_bone

        # Hindlimb
        hip_bone = edit_bones.new('Hip_L')
        hip_bone.head = spine_bones[0].head  # Pelvis
        hip_bone.tail = hip_bone.head + Vector((0, 0.05 * trunk_len, 0))

        thigh_bone = edit_bones.new('Thigh_L')
        thigh_bone.head = hip_bone.tail
        thigh_bone.tail = thigh_bone.head + Vector((0, self.cfg.armature.bone_lengths.hindlimb.hip_to_knee * trunk_len, 0))
        thigh_bone.parent = hip_bone

        shin_bone = edit_bones.new('Shin_L')
        shin_bone.head = thigh_bone.tail
        shin_bone.tail = shin_bone.head + Vector((0, self.cfg.armature.bone_lengths.hindlimb.knee_to_ankle * trunk_len, 0))
        shin_bone.parent = thigh_bone

        ankle_bone = edit_bones.new('Ankle_L')
        ankle_bone.head = shin_bone.tail
        ankle_bone.tail = ankle_bone.head + Vector((0, self.cfg.armature.bone_lengths.hindlimb.ankle_to_toe * trunk_len, 0))
        ankle_bone.parent = shin_bone

        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Mirror to create right side
        bpy.ops.object.mode_set(mode='EDIT')
        # Select left limbs
        for bone_name in ['Shoulder_L', 'Upper_Arm_L', 'Forearm_L', 'Wrist_L',
                          'Hip_L', 'Thigh_L', 'Shin_L', 'Ankle_L']:
            armature_data.edit_bones[bone_name].select = True

        # Symmetrize (this will create _R versions)
        bpy.ops.armature.symmetrize()
        bpy.ops.object.mode_set(mode='OBJECT')

        self.armature = armature_data
        print(f"[Armature] Created with {len(armature_data.bones)} bones")

    def apply_bone_transforms(self):
        """Apply rotations to bones for gliding pose"""
        # Enter pose mode
        bpy.context.view_layer.objects.active = self.armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        pose_bones = self.armature_obj.pose.bones

        # Apply forelimb spread
        spread_angle = self.cfg.armature.bone_angles.forelimb_spread
        for side in ['_L', '_R']:
            mult = 1 if side == '_L' else -1
            if f'Upper_Arm{side}' in pose_bones:
                pose_bones[f'Upper_Arm{side}'].rotation_mode = 'XYZ'
                pose_bones[f'Upper_Arm{side}'].rotation_euler[2] = mult * (spread_angle * 0.0174533)  # deg to rad

        # Apply hindlimb spread
        spread_angle = self.cfg.armature.bone_angles.hindlimb_spread
        for side in ['_L', '_R']:
            mult = 1 if side == '_L' else -1
            if f'Thigh{side}' in pose_bones:
                pose_bones[f'Thigh{side}'].rotation_mode = 'XYZ'
                pose_bones[f'Thigh{side}'].rotation_euler[2] = mult * (spread_angle * 0.0174533)

        # Apply pose as rest pose
        bpy.ops.pose.armature_apply(selected=False)
        bpy.ops.object.mode_set(mode='OBJECT')

        print("[Armature] Applied bone transforms")

    def add_appendages(self):
        """Add cartilage appendages (styloid, spurs)"""
        if not hasattr(self.cfg, 'appendages'):
            return

        bpy.ops.object.mode_set(mode='EDIT')
        edit_bones = self.armature.edit_bones
        trunk_len = self.cfg.armature.trunk_length

        # Wrist styloid
        if self.cfg.appendages.appendages.wrist_styloid.enabled:
            for side in ['_L', '_R']:
                parent_name = f'Wrist{side}'
                if parent_name in edit_bones:
                    styloid = edit_bones.new(f'Styloid{side}')
                    parent = edit_bones[parent_name]
                    styloid.head = parent.tail
                    length = self.cfg.appendages.appendages.wrist_styloid.length * trunk_len
                    # Extend along Y axis (outward)
                    mult = 1 if side == '_L' else -1
                    styloid.tail = styloid.head + Vector((0, mult * length, 0))
                    styloid.parent = parent
                    print(f"[Appendages] Added {styloid.name}")

        bpy.ops.object.mode_set(mode='OBJECT')

    def generate_body_mesh(self):
        """Generate body mesh using skin modifier on armature"""
        # For minimal test, create simple cylinders for body
        # In future, this will use skin modifier properly

        # Create a simple mesh for torso
        bpy.ops.mesh.primitive_cylinder_add(
            radius=self.cfg.body_mesh.torso.radius,
            depth=self.cfg.armature.trunk_length,
            location=(self.cfg.armature.trunk_length / 2, 0, 0),
            rotation=(0, 1.5708, 0)  # Rotate to align with X axis
        )
        torso = bpy.context.active_object
        torso.name = "Body"

        # Parent to armature
        torso.parent = self.armature_obj
        torso.parent_type = 'OBJECT'

        # Store reference
        self.mesh_obj = torso

        print("[Body] Created simple body mesh")

    def generate_patagium(self):
        """Generate patagium membrane using boundary fill method"""
        if not hasattr(self.cfg, 'patagium'):
            print("[Patagium] No patagium config, skipping")
            return

        # Get anchor point world positions
        self.compute_anchor_positions()

        for spline_name, spline_config in self.cfg.patagium.splines.items():
            # Skip disabled splines
            if 'enabled' in spline_config and not spline_config.enabled:
                continue

            print(f"[Patagium] Generating {spline_name}...")
            self.create_membrane_from_spline(spline_name, spline_config)

    def compute_anchor_positions(self):
        """Compute world positions of anchor points from bones"""
        bpy.context.view_layer.objects.active = self.armature_obj
        bpy.ops.object.mode_set(mode='OBJECT')

        for anchor_name, anchor_config in self.cfg.armature.anchor_points.items():
            bone_name = anchor_config.bone
            offset = Vector(anchor_config.offset)

            # Get bone tail position in world space
            if bone_name in self.armature.bones:
                bone = self.armature.bones[bone_name]
                # For left side
                bone_L_name = bone_name + "_L" if bone_name in ['Wrist', 'Ankle', 'Hip'] else bone_name
                if bone_L_name in self.armature.bones:
                    bone_pos = self.armature_obj.matrix_world @ self.armature.bones[bone_L_name].tail_local
                    self.anchor_points[anchor_name + "_L"] = bone_pos + offset
                    print(f"[Anchor] {anchor_name}_L at {bone_pos + offset}")

                # For right side
                bone_R_name = bone_name + "_R" if bone_name in ['Wrist', 'Ankle', 'Hip'] else bone_name
                if bone_R_name in self.armature.bones and bone_name != "Neck":  # Neck is centered
                    bone_pos = self.armature_obj.matrix_world @ self.armature.bones[bone_R_name].tail_local
                    offset_R = Vector((-offset.x, -offset.y, offset.z))  # Mirror X offset
                    self.anchor_points[anchor_name + "_R"] = bone_pos + offset_R
                    print(f"[Anchor] {anchor_name}_R at {bone_pos + offset_R}")

    def create_membrane_from_spline(self, spline_name: str, spline_config: DictConfig):
        """Create membrane mesh from spline config using boundary fill"""
        # Get anchor positions
        anchors = []
        for anchor_name in spline_config.anchors:
            # For left side
            full_name = anchor_name + "_L"
            if full_name in self.anchor_points:
                anchors.append(self.anchor_points[full_name])

        if len(anchors) < 2:
            print(f"[Patagium] Not enough anchors for {spline_name}, skipping")
            return

        # Create curve
        curve_data = bpy.data.curves.new(f'curve_{spline_name}', 'CURVE')
        curve_data.dimensions = '3D'

        spline = curve_data.splines.new('BEZIER')
        spline.bezier_points.add(len(anchors) - 1)

        # Set anchor points
        for i, anchor_pos in enumerate(anchors):
            point = spline.bezier_points[i]
            point.co = anchor_pos
            point.handle_left_type = 'AUTO'
            point.handle_right_type = 'AUTO'

        # Apply control points (offsets)
        if 'control_points' in spline_config:
            for ctrl_pt in spline_config.control_points:
                weight = ctrl_pt.weight
                offset = Vector(ctrl_pt.offset)
                # Interpolate position along spline
                idx = int(weight * (len(anchors) - 1))
                if idx < len(spline.bezier_points):
                    spline.bezier_points[idx].co += offset

        # Create curve object
        curve_obj = bpy.data.objects.new(f'Curve_{spline_name}', curve_data)
        bpy.context.collection.objects.link(curve_obj)

        # Convert curve to mesh for boundary fill
        # (Simplified for now - full implementation would do boundary fill)
        # For minimal test, just create a simple mesh between points
        mesh_data = bpy.data.meshes.new(f'mesh_{spline_name}')
        verts = [anchor_pos for anchor_pos in anchors]

        # Simple triangulation (for test)
        if len(verts) == 2:
            # Create a quad between two points (simplified membrane)
            width = 0.02
            v1 = verts[0]
            v2 = verts[1]
            perp = Vector((0, 0, 1))  # Up direction
            verts = [
                v1 + perp * width,
                v1 - perp * width,
                v2 - perp * width,
                v2 + perp * width
            ]
            faces = [(0, 1, 2, 3)]
            mesh_data.from_pydata(verts, [], faces)

        mesh_obj = bpy.data.objects.new(f'Membrane_{spline_name}', mesh_data)
        bpy.context.collection.objects.link(mesh_obj)

        # Parent to armature
        mesh_obj.parent = self.armature_obj

        print(f"[Patagium] Created {spline_name} membrane")

    def assign_material_regions(self):
        """Assign vertex groups for material regions"""
        # For minimal test, assign all vertices to 'membrane' group
        if self.mesh_obj:
            vg = self.mesh_obj.vertex_groups.new(name='body')
            vg.add(list(range(len(self.mesh_obj.data.vertices))), 1.0, 'REPLACE')
            print("[Regions] Assigned vertex groups")

    def triangulate_and_validate(self):
        """Triangulate all meshes and validate"""
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj != self.armature_obj:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
                bpy.ops.object.mode_set(mode='OBJECT')

        print("[Validation] Triangulated meshes")

    def export_files(self):
        """Export .blend, .obj, metadata, and planiform"""
        output_dir = Path(self.cfg.output.mesh_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = self.cfg.metadata.name

        # Export .blend file
        if self.cfg.output.blend_file:
            blend_path = output_dir / f"{base_name}.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
            print(f"[Export] Saved .blend: {blend_path}")

        # Export .obj file
        if self.cfg.output.obj_file:
            obj_path = output_dir / f"{base_name}.obj"
            # Select all mesh objects
            bpy.ops.object.select_all(action='DESELECT')
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    obj.select_set(True)

            bpy.ops.wm.obj_export(
                filepath=str(obj_path),
                export_selected_objects=True,
                apply_modifiers=True,
                export_triangulated_mesh=True
            )
            print(f"[Export] Saved .obj: {obj_path}")

        # Export metadata
        if self.cfg.output.metadata_file:
            metadata = {
                'name': self.cfg.metadata.name,
                'version': self.cfg.metadata.version,
                'trunk_length': self.cfg.armature.trunk_length,
                'anchor_points': {k: list(v) for k, v in self.anchor_points.items()},
                'material_regions': {
                    'body': list(range(len(self.mesh_obj.data.vertices))) if self.mesh_obj else []
                }
            }
            metadata_path = output_dir / f"{base_name}_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"[Export] Saved metadata: {metadata_path}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Hydra main entry point"""
    print(OmegaConf.to_yaml(cfg))

    generator = ParametricMeshGenerator(cfg)
    generator.generate()


if __name__ == "__main__":
    # Check if running in Blender
    if bpy.app.version_string:
        print(f"Running in Blender {bpy.app.version_string}")
        main()
    else:
        print("Error: This script must be run with Blender Python")
        sys.exit(1)
