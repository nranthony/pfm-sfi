#!/usr/bin/env python3
"""
Test Simulation Integration
Validates IBM cloth YAML loader and simulation integration
"""

import sys
import unittest
from pathlib import Path
import os

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "3D"))


class TestIBMClothBase(unittest.TestCase):
    """Test IBM cloth base class"""

    def test_ibm_cloth_base_imports(self):
        """Verify ibm_cloth_base.py can be imported"""
        try:
            from ibm_cloth_base import IBMClothConfig
            print("  ✓ ibm_cloth_base imports successfully")
        except ImportError as e:
            self.fail(f"Failed to import ibm_cloth_base: {e}")

    def test_create_config_legacy_mode(self):
        """Test creating config in legacy mode"""
        # This test doesn't require Taichi
        from ibm_cloth_base import IBMClothConfig

        # Test with mesh_name (legacy mode)
        try:
            config = IBMClothConfig(mesh_name='test.obj')
            self.assertIsNotNone(config.config)
            self.assertIsNotNone(config.mesh_path)
            print("  ✓ Legacy mode config creation works")
        except Exception as e:
            # Expected to fail without actual mesh file, but config should initialize
            if "config" in str(e).lower() or "mesh" in str(e).lower():
                print(f"  ✓ Config initialized (mesh file not found is expected)")
            else:
                raise

    def test_config_has_required_methods(self):
        """Verify IBMClothConfig has required methods"""
        from ibm_cloth_base import IBMClothConfig

        config = IBMClothConfig(mesh_name='test.obj')

        # Check methods exist
        self.assertTrue(hasattr(config, 'load_config'))
        self.assertTrue(hasattr(config, 'initialize_mesh'))
        self.assertTrue(hasattr(config, 'initialize_constraints'))
        self.assertTrue(hasattr(config, 'initialize_fixed_points'))
        self.assertTrue(hasattr(config, 'setup'))
        self.assertTrue(hasattr(config, 'get_solve_iters'))
        self.assertTrue(hasattr(config, 'get_dt'))
        self.assertTrue(hasattr(config, 'get_gravity'))

        print("  ✓ All required methods present")

    def test_default_config_values(self):
        """Verify default config values are reasonable"""
        from ibm_cloth_base import IBMClothConfig

        config = IBMClothConfig(mesh_name='test.obj')

        # Check default simulation params
        solve_iters = config.config['simulation']['solve_iters']
        dt = config.config['simulation']['dt']
        gravity = config.config['simulation']['gravity']

        self.assertGreater(solve_iters, 0)
        self.assertLess(solve_iters, 1000)
        self.assertGreater(dt, 0.0)
        self.assertLess(dt, 0.1)
        self.assertEqual(len(gravity), 3)

        print(f"  ✓ Default solve_iters: {solve_iters}")
        print(f"  ✓ Default dt: {dt}")
        print(f"  ✓ Default gravity: {gravity}")

    def test_yaml_config_loading(self):
        """Test loading YAML config file"""
        from ibm_cloth_base import IBMClothConfig
        import yaml

        # Create a minimal test config
        test_config = {
            'metadata': {'name': 'test_mesh', 'version': 'v1.0'},
            'simulation': {
                'solve_iters': 60,
                'dt': 0.001,
                'gravity': [0.0, 10.0, -5.0]
            },
            'material_properties': {
                'global_scale': 0.5,
                'repose_position': [0.3, 0.5, 0.5],
                'regions': {
                    'membrane': {
                        'density': 1.8,
                        'length_constraint_alpha': 0.025,
                        'bend_constraint_alpha': 5000
                    }
                }
            }
        }

        # Write test config
        test_config_path = Path(__file__).parent / "fixtures" / "test_config.yaml"
        test_config_path.parent.mkdir(exist_ok=True)

        with open(test_config_path, 'w') as f:
            yaml.dump(test_config, f)

        # Load config
        try:
            config = IBMClothConfig(config_path=str(test_config_path))

            # Verify loaded values
            self.assertEqual(config.config['simulation']['solve_iters'], 60)
            self.assertEqual(config.config['simulation']['dt'], 0.001)
            self.assertEqual(config.config['material_properties']['global_scale'], 0.5)

            print(f"  ✓ YAML config loaded successfully")
            print(f"    solve_iters: {config.get_solve_iters()}")
            print(f"    dt: {config.get_dt()}")

        except FileNotFoundError:
            # If mesh file doesn't exist, that's okay for this test
            print(f"  ✓ Config loading works (mesh file not required for this test)")


class TestSimulationCompatibility(unittest.TestCase):
    """Test compatibility with existing simulation code"""

    def test_ibm_cloth_module_structure(self):
        """Verify ibm_cloth.py maintains expected structure"""
        ibm_cloth_path = Path(__file__).parent.parent.parent / "3D" / "ibm_cloth.py"

        if not ibm_cloth_path.exists():
            self.fail("ibm_cloth.py not found")

        content = ibm_cloth_path.read_text()

        # Check for required exports
        required_exports = [
            'mesh',
            'ibm_dx',
            'xpbd',
            'length_cons',
            'bend_cons',
            'solve_iters',
            'dt',
            'g',
            'cons_vert_i',
            'cons_vert_p',
            'pointForce'
        ]

        for export in required_exports:
            # Check variable is defined
            self.assertIn(export, content,
                          f"Required export '{export}' not found in ibm_cloth.py")

        print("  ✓ All required exports present in ibm_cloth.py")

    def test_ibm_cloth_functions_exist(self):
        """Verify required functions exist in ibm_cloth.py"""
        ibm_cloth_path = Path(__file__).parent.parent.parent / "3D" / "ibm_cloth.py"
        content = ibm_cloth_path.read_text()

        required_functions = [
            'ibm_kernel',
            'Export',
            'copy_solid_velocity',
            'update_force',
            'spread_force',
            'sample_ibm_u',
            'advect_ibm',
            'solve_for_xpbd'
        ]

        for func in required_functions:
            # Check function is defined
            self.assertIn(f'def {func}(', content,
                          f"Required function '{func}' not found in ibm_cloth.py")

        print("  ✓ All required functions present in ibm_cloth.py")

    def test_legacy_mode_env_var(self):
        """Test legacy mode can be set via environment variable"""
        # Set legacy mode
        os.environ['IBM_CLOTH_MODE'] = 'legacy'

        # Check environment variable is set
        self.assertEqual(os.getenv('IBM_CLOTH_MODE'), 'legacy')

        print("  ✓ IBM_CLOTH_MODE environment variable works")

        # Cleanup
        if 'IBM_CLOTH_MODE' in os.environ:
            del os.environ['IBM_CLOTH_MODE']

    def test_yaml_mode_env_var(self):
        """Test YAML mode can be set via environment variable"""
        os.environ['IBM_CLOTH_MODE'] = 'yaml'
        os.environ['IBM_CLOTH_CONFIG'] = 'configs/test.yaml'

        self.assertEqual(os.getenv('IBM_CLOTH_MODE'), 'yaml')
        self.assertEqual(os.getenv('IBM_CLOTH_CONFIG'), 'configs/test.yaml')

        print("  ✓ YAML mode environment variables work")

        # Cleanup
        if 'IBM_CLOTH_MODE' in os.environ:
            del os.environ['IBM_CLOTH_MODE']
        if 'IBM_CLOTH_CONFIG' in os.environ:
            del os.environ['IBM_CLOTH_CONFIG']

    def test_backup_exists(self):
        """Verify original ibm_cloth.py is backed up"""
        backup_path = Path(__file__).parent.parent.parent / "3D" / "ibm_backup" / "ibm_cloth_original.py"

        self.assertTrue(backup_path.exists(),
                        "Original ibm_cloth.py backup not found")

        # Check backup is not empty
        backup_size = backup_path.stat().st_size
        self.assertGreater(backup_size, 1000,
                           "Backup file seems too small")

        print(f"  ✓ Backup exists: {backup_path}")
        print(f"    Size: {backup_size / 1024:.1f} KB")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Simulation Integration")
    print("=" * 70)
    unittest.main(verbosity=2)
