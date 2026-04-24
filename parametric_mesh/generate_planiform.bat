@echo off
REM Windows Batch Script to Generate SVG Planiform

echo ======================================================================
echo Parametric Mesh Planiform Generator (Windows)
echo ======================================================================
echo.

cd /d "%~dp0"

REM Check if mesh exists
if not exist "generated\meshes\flying_squirrel_minimal.blend" (
    echo ERROR: No mesh found. Please run generate_mesh.bat first.
    pause
    exit /b 1
)

REM Generate top view
echo Generating top view...
python scripts\blender_wrapper.py --script planiform_exporter.py -- --blend-file generated\meshes\flying_squirrel_minimal.blend --output generated\planforms\flying_squirrel_minimal_top.svg --view top --title "Flying Squirrel - Top View"

if errorlevel 1 (
    echo ERROR: Top view generation failed
    pause
    exit /b 1
)

REM Generate side view
echo.
echo Generating side view...
python scripts\blender_wrapper.py --script planiform_exporter.py -- --blend-file generated\meshes\flying_squirrel_minimal.blend --output generated\planforms\flying_squirrel_minimal_side.svg --view side --title "Flying Squirrel - Side View"

if errorlevel 1 (
    echo ERROR: Side view generation failed
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo Planiform Generation Complete
echo ======================================================================
echo.
echo Generated files:
dir /b generated\planforms\*.svg 2>nul
echo.
echo Open SVG files in a web browser to view
echo.
pause
