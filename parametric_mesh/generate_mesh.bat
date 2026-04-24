@echo off
REM Windows Batch Script to Generate Mesh
REM Uses blender_wrapper.py to handle Blender path

echo ======================================================================
echo Parametric Mesh Generator (Windows)
echo ======================================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.8+ or add to PATH
    pause
    exit /b 1
)

REM Run mesh generator via wrapper
echo Generating mesh...
echo.

python scripts\blender_wrapper.py --script mesh_generator.py %*

if errorlevel 1 (
    echo.
    echo ======================================================================
    echo ERROR: Mesh generation failed
    echo ======================================================================
    echo.
    echo Check the output above for errors.
    echo.
    echo Common issues:
    echo   1. Blender not configured - edit config\blender.yaml
    echo   2. Missing dependencies - run: pip install -r requirements.txt
    echo   3. Config errors - run: python tests\debug_config.py --all
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo Mesh Generation Complete
echo ======================================================================
echo.
echo Generated files in: generated\meshes\
dir /b generated\meshes\*.blend generated\meshes\*.obj 2>nul
echo.
echo Next steps:
echo   1. View mesh: "%BLENDER_PATH%" generated\meshes\flying_squirrel_minimal.blend
echo   2. Run tests: python tests\test_mesh_validation.py
echo   3. Generate planiform: generate_planiform.bat
echo.
pause
