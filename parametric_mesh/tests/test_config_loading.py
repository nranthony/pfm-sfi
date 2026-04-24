#!/usr/bin/env python3
"""
Test Configuration Loading
Validates Hydra config system and YAML parsing
"""

import sys
import unittest
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from omegaconf import DictConfig, OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    print("Warning: OmegaConf not available, skipping some tests")

import yaml


class TestConfigLoading(unittest.TestCase):
    """Test Hydra configuration loading"""

    def setUp(self):
        self.config_dir = Path(__file__).parent.parent / "config"

    def test_config_directory_exists(self):
        """Verify config directory exists"""
        self.assertTrue(self.config_dir.exists(), f"Config directory not found: {self.config_dir}")

    def test_base_config_loads(self):
        """Test base config.yaml loads without errors"""
        config_file = self.config_dir / "config.yaml"
        self.assertTrue(config_file.exists(), f"Base config not found: {config_file}")

        with open(config_file) as f:
            config = yaml.safe_load(f)

        self.assertIsNotNone(config)
        self.assertIn('metadata', config)
        self.assertIn('output', config)
        print(f"✓ Base config loaded: {config_file}")

    def test_armature_config_loads(self):
        """Test armature config loads"""
        armature_file = self.config_dir / "armature" / "base_rig.yaml"
        self.assertTrue(armature_file.exists(), f"Armature config not found: {armature_file}")

        with open(armature_file) as f:
            config = yaml.safe_load(f)

        self.assertIsNotNone(config)
        self.assertIn('trunk_length', config)
        self.assertIn('bone_lengths', config)
        self.assertIn('bone_angles', config)
        print(f"✓ Armature config loaded: {armature_file}")

    def test_species_config_loads(self):
        """Test species config loads"""
        species_file = self.config_dir / "species" / "flying_squirrel_minimal.yaml"
        self.assertTrue(species_file.exists(), f"Species config not found: {species_file}")

        with open(species_file) as f:
            config = yaml.safe_load(f)

        self.assertIsNotNone(config)
        self.assertIn('metadata', config)
        self.assertIn('patagium', config)
        print(f"✓ Species config loaded: {species_file}")

    def test_material_config_loads(self):
        """Test material config loads"""
        material_file = self.config_dir / "material" / "flexible_membrane.yaml"
        self.assertTrue(material_file.exists(), f"Material config not found: {material_file}")

        with open(material_file) as f:
            config = yaml.safe_load(f)

        self.assertIsNotNone(config)
        self.assertIn('material_properties', config)
        self.assertIn('regions', config['material_properties'])
        print(f"✓ Material config loaded: {material_file}")

    def test_appendages_config_loads(self):
        """Test appendages config loads"""
        appendages_file = self.config_dir / "appendages" / "squirrel_standard.yaml"
        self.assertTrue(appendages_file.exists(), f"Appendages config not found: {appendages_file}")

        with open(appendages_file) as f:
            config = yaml.safe_load(f)

        self.assertIsNotNone(config)
        self.assertIn('appendages', config)
        print(f"✓ Appendages config loaded: {appendages_file}")

    def test_bone_lengths_are_relative(self):
        """Verify bone lengths are relative proportions (< 1.5)"""
        armature_file = self.config_dir / "armature" / "base_rig.yaml"
        with open(armature_file) as f:
            config = yaml.safe_load(f)

        bone_lengths = config['bone_lengths']

        # Check spine
        for segment, length in bone_lengths['spine'].items():
            self.assertLess(length, 1.5,
                            f"Spine {segment} length {length} too large (should be < 1.5)")
            self.assertGreater(length, 0.0,
                               f"Spine {segment} length {length} must be positive")

        # Check limbs
        for limb_type in ['forelimb', 'hindlimb']:
            for bone, length in bone_lengths[limb_type].items():
                self.assertLess(length, 1.5,
                                f"{limb_type} {bone} length {length} too large")
                self.assertGreater(length, 0.0,
                                   f"{limb_type} {bone} length must be positive")

        print("✓ All bone lengths are valid relative proportions")

    def test_material_properties_valid(self):
        """Verify material properties have reasonable values"""
        material_file = self.config_dir / "material" / "flexible_membrane.yaml"
        with open(material_file) as f:
            config = yaml.safe_load(f)

        regions = config['material_properties']['regions']

        for region_name, props in regions.items():
            # Check density
            self.assertGreater(props['density'], 0.0,
                               f"{region_name} density must be positive")
            self.assertLess(props['density'], 10.0,
                            f"{region_name} density {props['density']} too high")

            # Check alpha values
            self.assertGreater(props['length_constraint_alpha'], 0.0,
                               f"{region_name} length_constraint_alpha must be positive")
            self.assertGreater(props['bend_constraint_alpha'], 0.0,
                               f"{region_name} bend_constraint_alpha must be positive")

        print("✓ All material properties are in valid ranges")

    def test_patagium_splines_valid(self):
        """Verify patagium spline definitions are valid"""
        species_file = self.config_dir / "species" / "flying_squirrel_minimal.yaml"
        with open(species_file) as f:
            config = yaml.safe_load(f)

        splines = config['patagium']['splines']

        for spline_name, spline_config in splines.items():
            # Skip disabled splines
            if 'enabled' in spline_config and not spline_config['enabled']:
                continue

            # Check required fields
            self.assertIn('type', spline_config, f"{spline_name} missing type")
            self.assertIn('anchors', spline_config, f"{spline_name} missing anchors")

            # Check anchors
            self.assertGreaterEqual(len(spline_config['anchors']), 2,
                                    f"{spline_name} needs at least 2 anchors")

            # Check control points if present
            if 'control_points' in spline_config:
                for i, ctrl_pt in enumerate(spline_config['control_points']):
                    self.assertIn('weight', ctrl_pt, f"{spline_name} control point {i} missing weight")
                    self.assertIn('offset', ctrl_pt, f"{spline_name} control point {i} missing offset")

                    # Weight should be 0-1
                    self.assertGreaterEqual(ctrl_pt['weight'], 0.0,
                                            f"{spline_name} control point {i} weight < 0")
                    self.assertLessEqual(ctrl_pt['weight'], 1.0,
                                         f"{spline_name} control point {i} weight > 1")

                    # Offset should be 3D
                    self.assertEqual(len(ctrl_pt['offset']), 3,
                                     f"{spline_name} control point {i} offset must be 3D")

        print("✓ All patagium splines are valid")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Configuration Loading")
    print("=" * 70)
    unittest.main(verbosity=2)
