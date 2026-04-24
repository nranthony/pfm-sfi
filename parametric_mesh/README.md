# Parametric Mesh Generation System

Automated generation of gliding animal meshes from configuration files for PFM-SFI simulations.

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r parametric_mesh/requirements.txt

# Ensure Blender is installed (3.6 LTS or 4.0+)
# Download from https://www.blender.org/download/
```

### 2. Generate Your First Mesh

```bash
# Navigate to parametric_mesh directory
cd parametric_mesh

# Generate minimal flying squirrel mesh
blender --background --python scripts/mesh_generator.py -- \
    --config-name=flying_squirrel_minimal

# Output files will be in:
# - generated/meshes/flying_squirrel_minimal.blend
# - generated/meshes/flying_squirrel_minimal.obj
# - generated/meshes/flying_squirrel_minimal_metadata.json
```

### 3. Export Planiform Visualization

```bash
# Generate top-down SVG view
blender --background --python scripts/planiform_exporter.py -- \
    --blend-file generated/meshes/flying_squirrel_minimal.blend \
    --output generated/planforms/flying_squirrel_minimal_top.svg \
    --view top

# Generate side view
blender --background --python scripts/planiform_exporter.py -- \
    --blend-file generated/meshes/flying_squirrel_minimal.blend \
    --output generated/planforms/flying_squirrel_minimal_side.svg \
    --view side
```

### 4. Run Simulation

```bash
# Copy generated mesh to simulation directory
cp generated/meshes/flying_squirrel_minimal.obj ../3D/assets/mesh/

# Copy metadata
cp generated/meshes/flying_squirrel_minimal_metadata.json ../3D/assets/mesh/

# Run simulation in YAML mode
cd ../3D
IBM_CLOTH_MODE=yaml IBM_CLOTH_CONFIG=configs/flying_squirrel_minimal.yaml python run.py
```

---

## Configuration System

### Directory Structure

```
parametric_mesh/
├── config/                         # Hydra configuration files
│   ├── config.yaml                # Base config
│   ├── armature/
│   │   └── base_rig.yaml         # Skeleton definition
│   ├── species/
│   │   └── flying_squirrel_minimal.yaml
│   ├── material/
│   │   └── flexible_membrane.yaml
│   └── appendages/
│       └── squirrel_standard.yaml
├── scripts/
│   ├── mesh_generator.py         # Main generation script
│   └── planiform_exporter.py     # SVG export script
├── generated/                     # Output directory
│   ├── meshes/
│   ├── planforms/
│   └── sim_configs/
└── README.md
```

### Configuration Files

#### 1. Armature Configuration (`config/armature/base_rig.yaml`)

Defines skeleton structure with bone lengths relative to trunk:

```yaml
trunk_length: 0.20  # Reference length

bone_lengths:
  spine:
    pelvis: 0.15      # Proportion of trunk_length
    lumbar: 0.20
    # ... more bones

  forelimb:
    shoulder_to_elbow: 0.75
    elbow_to_wrist: 0.70

bone_angles:
  forelimb_spread: 75.0  # Degrees
  hindlimb_spread: 80.0
```

#### 2. Species Configuration (`config/species/flying_squirrel_minimal.yaml`)

Defines patagium splines and mesh properties:

```yaml
metadata:
  name: "flying_squirrel_minimal"
  species: "Glaucomys volans"

patagium:
  splines:
    main_patagium:
      type: "bezier"
      anchors: ["wrist", "ankle"]
      control_points:
        - weight: 0.5
          offset: [0.0, 0.08, 0.0]  # Membrane curvature
      subdivision: 20
```

#### 3. Material Configuration (`config/material/flexible_membrane.yaml`)

Maps to IBM cloth simulation parameters:

```yaml
material_properties:
  global_scale: 0.5
  repose_position: [0.3, 0.5, 0.5]

  regions:
    membrane:
      density: 1.5
      length_constraint_alpha: 0.02   # Elasticity
      bend_constraint_alpha: 4000     # Stiffness
```

#### 4. Appendages Configuration (`config/appendages/squirrel_standard.yaml`)

Optional cartilage extensions:

```yaml
appendages:
  wrist_styloid:
    enabled: true
    length: 0.15      # Relative to trunk_length
    angle: 15.0       # Degrees

  ankle_spur:
    enabled: false
```

---

## Workflow

### Human-in-the-Loop Refinement

1. **Generate initial mesh** with Blender script
2. **Open .blend file** in Blender GUI
3. **Inspect and modify:**
   - View vertex groups (membrane, body, limbs)
   - Adjust bone positions manually
   - Paint skin weights if needed
   - Modify patagium mesh manually
4. **Re-export .obj** from Blender
5. **Run simulation** with updated mesh

### Viewing Vertex Groups in Blender

```
1. Open generated .blend file
2. Select mesh object
3. Open Properties panel → Object Data → Vertex Groups
4. Select group (e.g., "membrane", "body")
5. Enable "Weight Paint" mode to visualize
```

---

## Parameter Sweeps with Hydra

### Override Parameters

```bash
# Change bone length
blender --background --python scripts/mesh_generator.py -- \
    --config-name=flying_squirrel_minimal \
    armature.trunk_length=0.25

# Change membrane curvature
blender --background --python scripts/mesh_generator.py -- \
    --config-name=flying_squirrel_minimal \
    +patagium.splines.main_patagium.control_points[0].offset=[0.0,0.10,0.0]
```

### Multi-Run Sweeps

```bash
# Sweep multiple trunk lengths
blender --background --python scripts/mesh_generator.py -- \
    --multirun \
    --config-name=flying_squirrel_minimal \
    armature.trunk_length=0.15,0.20,0.25,0.30
```

---

## Integration with Simulation

### YAML Mode (Recommended)

The simulation can load configs directly:

```python
# In 3D/ibm_cloth.py
MODE = 'yaml'  # Set to 'yaml'
CONFIG_PATH = 'configs/flying_squirrel_minimal.yaml'
```

Or via environment variable:

```bash
export IBM_CLOTH_MODE=yaml
export IBM_CLOTH_CONFIG=configs/flying_squirrel_minimal.yaml
cd 3D && python run.py
```

### Legacy Mode (Backward Compatible)

```python
# In 3D/ibm_cloth.py
MODE = 'legacy'  # Uses hardcoded silk2.obj
```

---

## File Outputs

### Mesh Files

- **`.blend`**: Full Blender file with armature + mesh (for refinement)
- **`.obj`**: Triangulated mesh for simulation
- **`_metadata.json`**: Vertex groups, anchor points, material regions

### Planiform Files

- **`_top.svg`**: Top-down view (XY plane)
- **`_side.svg`**: Side view (XZ plane)
- **`_front.svg`**: Front view (YZ plane)

---

## Troubleshooting

### Blender Not Found

```bash
# Add Blender to PATH or use full path
/Applications/Blender.app/Contents/MacOS/Blender --background --python ...
```

### Hydra Config Errors

```bash
# Check config structure
python -c "from omegaconf import OmegaConf; print(OmegaConf.to_yaml(OmegaConf.load('config/config.yaml')))"
```

### Mesh Generation Fails

1. Check Blender version (>= 3.6)
2. Ensure all config files exist
3. Check Blender console output for errors

### Simulation Crashes

1. Verify mesh is triangulated (open in Blender, check faces)
2. Check mesh is manifold (no holes)
3. Reduce grid resolution in hyperparameters.py for testing

---

## Next Steps

1. **Create custom species**: Copy `flying_squirrel_minimal.yaml` and modify
2. **Tune material properties**: Adjust alpha values in material configs
3. **Run parameter sweeps**: Use Hydra multirun for optimization
4. **Refine meshes manually**: Open .blend files and adjust in Blender GUI

---

## Reference

- Project SSoT (mesh requirements, simulation setup, staged ladder, curvature strategy): `../FLYING_SQUIRREL_GUIDE.md`
- Agent-facing repo orientation: `../CLAUDE.md`
