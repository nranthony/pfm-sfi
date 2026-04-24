# Parametric Mesh Generation - Windows Setup Guide

Quick setup guide for Windows users.

## Prerequisites

1. **Python 3.8+** - [Download from python.org](https://www.python.org/downloads/)
2. **Blender 4.5.5** - Already installed at:
   ```
   C:\Blender Foundation\stable\blender-4.5.5-windows-x64\blender.exe
   ```

## One-Time Setup

### Step 1: Configure Blender Path

The Blender path is already configured in `config/blender.yaml`:

```yaml
windows_path: "C:\\Blender Foundation\\stable\\blender-4.5.5-windows-x64\\blender.exe"
```

✅ No changes needed if Blender is at the default location above.

If your Blender is elsewhere, edit `config/blender.yaml` and update `windows_path`.

### Step 2: Install Python Dependencies

Open Command Prompt in the `parametric_mesh` directory and run:

```cmd
pip install -r requirements.txt
```

This installs:
- hydra-core (config system)
- omegaconf (config parsing)
- svgwrite (SVG export)
- pyyaml (YAML parsing)

## Usage

### Quick Start: Generate Your First Mesh

**Option 1: Using Batch Script (Easiest)**

Double-click `generate_mesh.bat` or run:

```cmd
generate_mesh.bat
```

This will:
1. Find Blender using the config
2. Generate the mesh
3. Show you the output files

**Option 2: Using Python Wrapper**

```cmd
python scripts\blender_wrapper.py --script mesh_generator.py
```

**Option 3: Direct Blender Call**

```cmd
"C:\Blender Foundation\stable\blender-4.5.5-windows-x64\blender.exe" --background --python scripts\mesh_generator.py
```

### Generate SVG Planiform

After generating a mesh:

```cmd
generate_planiform.bat
```

Or manually:

```cmd
python scripts\blender_wrapper.py --script planiform_exporter.py -- --blend-file generated\meshes\flying_squirrel_minimal.blend --output generated\planforms\test_top.svg --view top
```

### View Generated Mesh in Blender

```cmd
"C:\Blender Foundation\stable\blender-4.5.5-windows-x64\blender.exe" generated\meshes\flying_squirrel_minimal.blend
```

Or set the environment variable and use:

```cmd
set BLENDER_PATH=C:\Blender Foundation\stable\blender-4.5.5-windows-x64\blender.exe
"%BLENDER_PATH%" generated\meshes\flying_squirrel_minimal.blend
```

## Blender Path Configuration Methods

The system supports 3 ways to specify Blender path (in order of precedence):

### Method 1: Environment Variable (Temporary)

Set for current session:

```cmd
set BLENDER_PATH=C:\Blender Foundation\stable\blender-4.5.5-windows-x64\blender.exe
```

Set permanently:

```cmd
setx BLENDER_PATH "C:\Blender Foundation\stable\blender-4.5.5-windows-x64\blender.exe"
```

### Method 2: Config File (Recommended)

Edit `config/blender.yaml`:

```yaml
windows_path: "C:\\Blender Foundation\\stable\\blender-4.5.5-windows-x64\\blender.exe"
```

✅ Already configured correctly for your system!

### Method 3: Command-Line Argument

```cmd
python scripts\blender_wrapper.py --blender-path "C:\path\to\blender.exe" --script mesh_generator.py
```

## Running Tests

### Test Configuration (No Blender Needed)

```cmd
python tests\test_config_loading.py
python tests\test_simulation_integration.py
```

### Debug Configuration

```cmd
python tests\debug_config.py --all
```

### Test Blender Environment

```cmd
python scripts\blender_wrapper.py --script ..\tests\debug_blender.py
```

### Full End-to-End Test

```cmd
REM Install bash for Windows (Git Bash or WSL) or run individual commands
tests\test_end_to_end.sh
```

## Common Issues

### Issue: "Blender executable not found"

**Solution:**

Check your config:

```cmd
type config\blender.yaml
```

Make sure `windows_path` matches your Blender location.

### Issue: "Python not found"

**Solution:**

Add Python to PATH during installation, or run:

```cmd
py scripts\blender_wrapper.py --script mesh_generator.py
```

### Issue: "Module not found"

**Solution:**

Install dependencies:

```cmd
pip install -r requirements.txt
```

### Issue: Blender opens in GUI mode

**Solution:**

Make sure you're using the wrapper script or include `--background`:

```cmd
python scripts\blender_wrapper.py --script mesh_generator.py
```

## File Locations

**Generated Files:**
```
parametric_mesh\
├── generated\
│   ├── meshes\
│   │   ├── flying_squirrel_minimal.blend
│   │   ├── flying_squirrel_minimal.obj
│   │   └── flying_squirrel_minimal_metadata.json
│   └── planforms\
│       ├── flying_squirrel_minimal_top.svg
│       └── flying_squirrel_minimal_side.svg
```

**Configuration:**
```
parametric_mesh\
├── config\
│   ├── blender.yaml          ← Blender path here
│   ├── armature\base_rig.yaml
│   ├── species\flying_squirrel_minimal.yaml
│   └── ...
```

## Workflow Summary

1. **Configure Blender path** (already done ✓)
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Generate mesh:** `generate_mesh.bat`
4. **View in Blender:** Open `.blend` file
5. **Generate planiform:** `generate_planiform.bat`
6. **Refine manually:** Edit `.blend` in Blender GUI
7. **Export for simulation:** Files auto-copied to `3D\assets\mesh\`

## Next Steps

1. Run `generate_mesh.bat` to test the system
2. Open generated `.blend` file in Blender
3. Modify configs in `config\` directory
4. Re-generate and iterate

---

**See also:**
- `README.md` - Full system documentation
- `..\FLYING_SQUIRREL_GUIDE.md` - Project SSoT
- `tests\README.md` - Testing guide
