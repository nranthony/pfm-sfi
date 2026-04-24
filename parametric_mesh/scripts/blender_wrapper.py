#!/usr/bin/env python3
"""
Blender Wrapper
Handles cross-platform Blender execution with proper path resolution

Usage:
    python blender_wrapper.py --script mesh_generator.py [-- script args]
    python blender_wrapper.py --script planiform_exporter.py -- --output test.svg
"""

import sys
import os
import subprocess
import platform
from pathlib import Path
import argparse

# Add parametric_mesh to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    print("Warning: OmegaConf not available, will use environment variables or PATH")


def find_blender_executable():
    """
    Find Blender executable in order of precedence:
    1. BLENDER_PATH environment variable
    2. Hydra config (blender.yaml)
    3. System PATH
    """

    # 1. Check environment variable
    env_path = os.getenv('BLENDER_PATH')
    if env_path and Path(env_path).exists():
        print(f"[Blender] Using BLENDER_PATH: {env_path}")
        return env_path

    # 2. Check Hydra config
    if OMEGACONF_AVAILABLE:
        try:
            config_path = Path(__file__).parent.parent / "config" / "blender.yaml"
            if config_path.exists():
                cfg = OmegaConf.load(config_path)

                # Get platform-specific path
                system = platform.system().lower()
                if system == 'windows' and cfg.get('windows_path'):
                    blender_path = cfg.windows_path
                    if Path(blender_path).exists():
                        print(f"[Blender] Using config windows_path: {blender_path}")
                        return blender_path
                elif system == 'linux' and cfg.get('linux_path'):
                    blender_path = cfg.linux_path
                    if Path(blender_path).exists():
                        print(f"[Blender] Using config linux_path: {blender_path}")
                        return blender_path
                elif system == 'darwin' and cfg.get('macos_path'):
                    blender_path = cfg.macos_path
                    if Path(blender_path).exists():
                        print(f"[Blender] Using config macos_path: {blender_path}")
                        return blender_path
        except Exception as e:
            print(f"[Blender] Warning: Could not load config: {e}")

    # 3. Check system PATH
    blender_cmd = 'blender'
    if platform.system() == 'Windows':
        blender_cmd = 'blender.exe'

    # Try to find in PATH
    from shutil import which
    path_blender = which(blender_cmd)
    if path_blender:
        print(f"[Blender] Using PATH: {path_blender}")
        return path_blender

    # Not found
    return None


def get_blender_version(blender_path):
    """Get Blender version"""
    try:
        result = subprocess.run(
            [blender_path, '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Parse version from output
        for line in result.stdout.split('\n'):
            if 'Blender' in line:
                return line.strip()

        return "Unknown version"
    except Exception as e:
        return f"Could not determine version: {e}"


def run_blender_script(blender_path, script_path, script_args=None, background=True):
    """
    Run Blender script

    Args:
        blender_path: Path to Blender executable
        script_path: Path to Python script to run in Blender
        script_args: List of arguments to pass to script (after --)
        background: Run in background mode
    """

    cmd = [blender_path]

    if background:
        cmd.append('--background')

    cmd.extend(['--python', str(script_path)])

    if script_args:
        cmd.append('--')
        cmd.extend(script_args)

    print(f"[Blender] Command: {' '.join(cmd)}")
    print(f"[Blender] Running script: {script_path}")
    print("-" * 70)

    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 70)
        print(f"[Blender] Script completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print("-" * 70)
        print(f"[Blender] Script failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n[Blender] Interrupted by user")
        return 130


def main():
    parser = argparse.ArgumentParser(description='Blender wrapper script')
    parser.add_argument('--script', type=str, required=True,
                        help='Python script to run in Blender (e.g., mesh_generator.py)')
    parser.add_argument('--blender-path', type=str,
                        help='Override Blender executable path')
    parser.add_argument('--foreground', action='store_true',
                        help='Run Blender in foreground (with GUI)')

    # Parse known args, rest go to script
    args, script_args = parser.parse_known_args()

    # Remove '--' if present at start of script_args
    if script_args and script_args[0] == '--':
        script_args = script_args[1:]

    # Find Blender
    if args.blender_path:
        blender_path = args.blender_path
        print(f"[Blender] Using provided path: {blender_path}")
    else:
        blender_path = find_blender_executable()

    if not blender_path:
        print("ERROR: Blender executable not found!")
        print("\nPlease do one of the following:")
        print("  1. Set BLENDER_PATH environment variable:")
        print("     export BLENDER_PATH='/path/to/blender'  # Linux/Mac")
        print("     set BLENDER_PATH=C:\\path\\to\\blender.exe  # Windows")
        print()
        print("  2. Edit config/blender.yaml and set windows_path/linux_path/macos_path")
        print()
        print("  3. Add Blender to system PATH")
        print()
        print("  4. Use --blender-path argument:")
        print("     python blender_wrapper.py --blender-path /path/to/blender --script ...")
        return 1

    # Check Blender exists
    if not Path(blender_path).exists():
        print(f"ERROR: Blender not found at: {blender_path}")
        return 1

    # Get version
    version = get_blender_version(blender_path)
    print(f"[Blender] {version}")

    # Resolve script path
    script_path = Path(__file__).parent / args.script
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return 1

    # Run script
    background = not args.foreground
    return run_blender_script(blender_path, script_path, script_args, background)


if __name__ == '__main__':
    sys.exit(main())
