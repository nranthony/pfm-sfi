# Blender Path Configuration Guide

Your Blender is already configured! Here's how the system finds Blender and what options you have.

## Current Configuration

✅ **Blender 4.5.5** configured in `config/blender.yaml`:

```yaml
windows_path: "C:\\Blender Foundation\\stable\\blender-4.5.5-windows-x64\\blender.exe"
```

## How It Works

The system finds Blender in this order:

1. **`BLENDER_PATH` environment variable** (if set)
2. **Config file** (`config/blender.yaml`) ✅ *You're using this*
3. **System PATH** (if Blender is in PATH)

## Quick Test

Verify your Blender configuration:

```cmd
cd parametric_mesh
python scripts\blender_wrapper.py --script mesh_generator.py
```

This will:
- ✓ Find Blender using `config/blender.yaml`
- ✓ Show Blender version
- ✓ Generate test mesh

## Three Ways to Configure

### Option 1: Config File (Recommended - Already Set Up!)

**Location:** `config/blender.yaml`

```yaml
windows_path: "C:\\Blender Foundation\\stable\\blender-4.5.5-windows-x64\\blender.exe"
```

**Pros:**
- ✅ Persists across sessions
- ✅ Version controlled (team can share)
- ✅ Platform-specific (works on Windows, Linux, Mac)
- ✅ Already configured for you!

**Cons:**
- Must use double backslashes (`\\`) on Windows

**When to use:** Default choice, already done for you!

---

### Option 2: Environment Variable (Override)

**Temporary (current session only):**

```cmd
set BLENDER_PATH=C:\Blender Foundation\stable\blender-4.5.5-windows-x64\blender.exe
python scripts\blender_wrapper.py --script mesh_generator.py
```

**Permanent (all sessions):**

```cmd
setx BLENDER_PATH "C:\Blender Foundation\stable\blender-4.5.5-windows-x64\blender.exe"
```

Then restart your terminal.

**Pros:**
- ✅ Overrides config file
- ✅ Easy to change temporarily
- ✅ Single backslashes work

**Cons:**
- Not version controlled
- Must set on each machine

**When to use:** Testing different Blender versions or machine-specific override

---

### Option 3: Command-Line Argument (One-Off)

```cmd
python scripts\blender_wrapper.py --blender-path "C:\path\to\blender.exe" --script mesh_generator.py
```

**Pros:**
- ✅ Quick override for testing
- ✅ No config changes needed

**Cons:**
- Must type every time
- Not persistent

**When to use:** Quick testing or CI/CD with dynamic paths

---

## Easy Usage (With Config Already Set)

Since your Blender path is already in `config/blender.yaml`, just use:

### Generate Mesh

**Windows batch script:**
```cmd
generate_mesh.bat
```

**Python wrapper:**
```cmd
python scripts\blender_wrapper.py --script mesh_generator.py
```

### Generate Planiform

```cmd
generate_planiform.bat
```

Or:

```cmd
python scripts\blender_wrapper.py --script planiform_exporter.py -- --blend-file generated\meshes\flying_squirrel_minimal.blend --output test.svg --view top
```

### Run Tests

```cmd
python scripts\blender_wrapper.py --script ..\tests\debug_blender.py
```

## Updating Blender Version

If you install a different Blender version later:

### Quick Update

Edit `config/blender.yaml`:

```yaml
# Change this line:
windows_path: "C:\\Blender Foundation\\stable\\blender-4.6.0-windows-x64\\blender.exe"
```

### Or Use Environment Variable

```cmd
setx BLENDER_PATH "C:\path\to\new\blender.exe"
```

## Multi-Platform Support

The config supports all platforms:

```yaml
# config/blender.yaml

windows_path: "C:\\Blender Foundation\\stable\\blender-4.5.5-windows-x64\\blender.exe"
linux_path: "/usr/bin/blender"
macos_path: "/Applications/Blender.app/Contents/MacOS/Blender"
```

The wrapper automatically picks the right one for your OS.

## Troubleshooting

### "Blender executable not found"

1. Check config file:
   ```cmd
   type config\blender.yaml
   ```

2. Verify Blender exists:
   ```cmd
   dir "C:\Blender Foundation\stable\blender-4.5.5-windows-x64\blender.exe"
   ```

3. Update path if needed

### "Wrong Blender version"

Check version:
```cmd
"C:\Blender Foundation\stable\blender-4.5.5-windows-x64\blender.exe" --version
```

Should see: `Blender 4.5.5`

### "Blender opens in GUI"

Make sure you're using the wrapper or batch scripts - they automatically add `--background`.

Don't use Blender directly unless you want the GUI.

## Summary

✅ **Your system is ready!**

Blender path already configured in `config/blender.yaml`

Just run:
```cmd
generate_mesh.bat
```

No additional setup needed!

---

**See also:**
- `README_WINDOWS.md` - Full Windows setup guide
- `README.md` - Complete system documentation
