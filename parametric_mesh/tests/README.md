# Parametric Mesh Tests

Comprehensive test suite for debugging and validating the parametric mesh generation system.

## Test Structure

```
tests/
├── unit/                          # Unit tests (future)
├── integration/                   # Integration tests (future)
├── fixtures/                      # Test data and configs
├── outputs/                       # Test outputs
├── __init__.py
├── test_config_loading.py         # Config validation tests
├── test_mesh_validation.py        # Mesh quality tests
├── test_simulation_integration.py # Simulation integration tests
├── test_end_to_end.sh            # Full workflow test (bash)
├── debug_config.py                # Config debugging utility
├── debug_blender.py               # Blender debugging utility
└── README.md                      # This file
```

## Running Tests

### Quick Test (Config Only)

Test configuration files without Blender:

```bash
cd parametric_mesh

# Test config loading
python tests/test_config_loading.py

# Test simulation integration
python tests/test_simulation_integration.py

# Debug specific config file
python tests/debug_config.py --file config/armature/base_rig.yaml

# Debug all configs
python tests/debug_config.py --all
```

### Full End-to-End Test

Test complete workflow including Blender mesh generation:

```bash
cd parametric_mesh

# Run full workflow test
./tests/test_end_to_end.sh
```

This will:
1. Validate configurations
2. Generate mesh with Blender
3. Validate generated mesh
4. Generate SVG planiform
5. Test simulation integration
6. Copy files to simulation directory

### Blender-Specific Debugging

Debug Blender operations and mesh generation:

```bash
cd parametric_mesh

# Run Blender debug tests
blender --background --python tests/debug_blender.py

# Check outputs
ls tests/outputs/
# → debug_test.blend
# → debug_test.obj
```

### Mesh Validation

Validate generated meshes after generation:

```bash
cd parametric_mesh

# First, generate a mesh
blender --background --python scripts/mesh_generator.py

# Then validate it
python tests/test_mesh_validation.py
```

## Test Coverage

### Configuration Tests (`test_config_loading.py`)

✅ Config directory structure
✅ YAML file loading (PyYAML)
✅ Hydra/OmegaConf compatibility
✅ Bone length validation (relative proportions)
✅ Material property ranges
✅ Patagium spline definitions

**Run:** `python tests/test_config_loading.py`

### Mesh Validation Tests (`test_mesh_validation.py`)

✅ Mesh triangulation (only triangles)
✅ Vertex count (reasonable range)
✅ No degenerate triangles (zero area)
✅ Mesh bounds (not too large/small)
✅ Metadata JSON export
✅ Blender file (.blend) export
✅ OBJ file (.obj) export
✅ SVG planiform export

**Run:** `python tests/test_mesh_validation.py`

**Note:** Requires generated meshes. Run `mesh_generator.py` first.

### Simulation Integration Tests (`test_simulation_integration.py`)

✅ `ibm_cloth_base.py` imports
✅ IBMClothConfig class structure
✅ Default parameter values
✅ YAML config loading
✅ Legacy mode compatibility
✅ Required exports (mesh, xpbd, etc.)
✅ Required functions (ibm_kernel, etc.)
✅ Environment variable control
✅ Backup file exists

**Run:** `python tests/test_simulation_integration.py`

### Blender Debug Tests (`debug_blender.py`)

✅ Blender environment check
✅ Basic operations (create objects)
✅ Armature generation (bones)
✅ Mesh creation and triangulation
✅ File export (.blend, .obj)

**Run:** `blender --background --python tests/debug_blender.py`

### Config Debug Utility (`debug_config.py`)

Utilities for inspecting configurations:

```bash
# Check all configs
python tests/debug_config.py --all

# Debug specific file
python tests/debug_config.py --file config/species/flying_squirrel_minimal.yaml
```

**Features:**
- YAML syntax validation
- Structure inspection
- Bone length validation
- Material property validation
- Completeness check

## Common Issues and Solutions

### Issue: "Blender not found"

**Solution:**
```bash
# Add Blender to PATH or use full path
export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"  # macOS
# or
/path/to/blender --background --python ...
```

### Issue: "meshio not installed"

**Solution:**
```bash
pip install meshio
```

### Issue: "Config file not found"

**Solution:**
Check you're in the correct directory:
```bash
cd parametric_mesh
python tests/test_config_loading.py
```

### Issue: "No generated meshes found"

**Solution:**
Generate a mesh first:
```bash
blender --background --python scripts/mesh_generator.py
```

### Issue: "YAML parse error"

**Solution:**
Use debug utility to find the issue:
```bash
python tests/debug_config.py --file path/to/config.yaml
```

### Issue: Tests fail with "No module named 'taichi'"

**Solution:**
Simulation integration tests don't actually need Taichi - they test the structure.
If you want to run simulations, install dependencies:
```bash
cd ..
pip install -r requirements.txt
```

## Test Output Interpretation

### Successful Test Output

```
✓ Config loaded successfully
✓ All bone lengths are valid
✓ Mesh triangulated
✓ No degenerate triangles
✓ Bounds are reasonable
```

### Failed Test Output

```
❌ Config file not found
❌ Bone length out of range
❌ Mesh contains quads (should be triangles)
❌ Found degenerate triangles
```

### Warning Output

```
⚠ No .blend file found (may not have run generator yet)
⚠ Metadata JSON not generated (optional)
```

Warnings indicate non-critical issues or optional features.

## Continuous Testing During Development

### Quick Check Workflow

```bash
# 1. Modify config
vim config/species/flying_squirrel_minimal.yaml

# 2. Validate config
python tests/debug_config.py --file config/species/flying_squirrel_minimal.yaml

# 3. Generate mesh
blender --background --python scripts/mesh_generator.py

# 4. Validate mesh
python tests/test_mesh_validation.py

# 5. View in Blender
blender generated/meshes/flying_squirrel_minimal.blend
```

### Full Validation

```bash
# Run complete test suite
./tests/test_end_to_end.sh
```

## Test Data

### Fixtures

Test fixtures (example configs, reference meshes) go in `tests/fixtures/`:

```
tests/fixtures/
├── test_config.yaml       # Minimal test config
├── simple_mesh.obj        # Reference mesh for comparison
└── expected_output.json   # Expected metadata structure
```

### Outputs

Test outputs go in `tests/outputs/`:

```
tests/outputs/
├── debug_test.blend       # Debug Blender file
├── debug_test.obj         # Debug OBJ export
└── test_logs/             # Test execution logs
```

**Note:** `tests/outputs/` is gitignored.

## Writing New Tests

### Unit Test Template

```python
#!/usr/bin/env python3
import unittest
from pathlib import Path

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Setup code
        pass

    def test_feature(self):
        # Test code
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

### Blender Test Template

```python
#!/usr/bin/env python3
import bpy

def test_blender_feature():
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Test code
    assert len(bpy.data.objects) == 0

if __name__ == '__main__':
    test_blender_feature()
    print("✓ Test passed")
```

## CI/CD Integration (Future)

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Test Configurations
  run: |
    cd parametric_mesh
    python tests/test_config_loading.py
    python tests/test_simulation_integration.py

- name: Test Mesh Generation
  run: |
    cd parametric_mesh
    blender --background --python scripts/mesh_generator.py
    python tests/test_mesh_validation.py
```

## Test Coverage Goals

- [ ] Config loading: 100%
- [x] Mesh validation: 80% (requires generation)
- [x] Simulation integration: 90%
- [ ] Blender operations: 60%
- [ ] End-to-end workflow: 70%

## Contributing Tests

When adding new features, please add corresponding tests:

1. **Config changes** → Update `test_config_loading.py`
2. **Mesh generation** → Update `test_mesh_validation.py`
3. **Simulation changes** → Update `test_simulation_integration.py`
4. **Blender features** → Update `debug_blender.py`

---

**Questions?** See `../../FLYING_SQUIRREL_GUIDE.md` for usage guidance.
