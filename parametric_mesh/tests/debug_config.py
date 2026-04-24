#!/usr/bin/env python3
"""
Debug Configuration Issues
Utility to inspect and validate configuration files
"""

import sys
from pathlib import Path
import yaml
from pprint import pprint

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    print("Warning: OmegaConf not available")


def debug_yaml_file(yaml_path):
    """Load and display YAML file with error handling"""
    print(f"\n{'='*70}")
    print(f"Debugging: {yaml_path}")
    print(f"{'='*70}\n")

    yaml_path = Path(yaml_path)

    # Check file exists
    if not yaml_path.exists():
        print(f"❌ File not found: {yaml_path}")
        return False

    print(f"✓ File exists: {yaml_path}")
    print(f"  Size: {yaml_path.stat().st_size} bytes")

    # Try to load with PyYAML
    print("\n--- Loading with PyYAML ---")
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        print("✓ PyYAML load successful")
        print("\nStructure:")
        pprint(data, depth=2)
    except yaml.YAMLError as e:
        print(f"❌ PyYAML parse error:")
        print(f"  {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error:")
        print(f"  {e}")
        return False

    # Try to load with OmegaConf if available
    if OMEGACONF_AVAILABLE:
        print("\n--- Loading with OmegaConf ---")
        try:
            cfg = OmegaConf.load(yaml_path)
            print("✓ OmegaConf load successful")
            print("\nOmegaConf YAML:")
            print(OmegaConf.to_yaml(cfg))
        except Exception as e:
            print(f"❌ OmegaConf error:")
            print(f"  {e}")
            return False

    return True


def check_config_completeness():
    """Check all required config files exist"""
    print(f"\n{'='*70}")
    print("Checking Configuration Completeness")
    print(f"{'='*70}\n")

    config_dir = Path(__file__).parent.parent / "config"

    required_files = {
        "Base config": config_dir / "config.yaml",
        "Armature config": config_dir / "armature" / "base_rig.yaml",
        "Species config": config_dir / "species" / "flying_squirrel_minimal.yaml",
        "Material config": config_dir / "material" / "flexible_membrane.yaml",
        "Appendages config": config_dir / "appendages" / "squirrel_standard.yaml",
    }

    all_present = True
    for name, path in required_files.items():
        if path.exists():
            print(f"✓ {name}: {path.name}")
        else:
            print(f"❌ {name}: MISSING - {path}")
            all_present = False

    return all_present


def validate_bone_lengths():
    """Validate bone length configurations"""
    print(f"\n{'='*70}")
    print("Validating Bone Lengths")
    print(f"{'='*70}\n")

    armature_file = Path(__file__).parent.parent / "config" / "armature" / "base_rig.yaml"

    if not armature_file.exists():
        print("❌ Armature config not found")
        return False

    with open(armature_file) as f:
        config = yaml.safe_load(f)

    trunk_length = config.get('trunk_length', 0)
    print(f"Trunk length: {trunk_length}")

    bone_lengths = config.get('bone_lengths', {})

    issues = []

    # Check spine
    if 'spine' in bone_lengths:
        print("\nSpine segments:")
        for segment, length in bone_lengths['spine'].items():
            abs_length = length * trunk_length
            status = "✓" if 0 < length < 1.5 else "❌"
            print(f"  {status} {segment}: {length:.3f} (abs: {abs_length:.3f})")
            if not (0 < length < 1.5):
                issues.append(f"{segment} length {length} out of range")

    # Check limbs
    for limb_type in ['forelimb', 'hindlimb']:
        if limb_type in bone_lengths:
            print(f"\n{limb_type.capitalize()}:")
            for bone, length in bone_lengths[limb_type].items():
                abs_length = length * trunk_length
                status = "✓" if 0 < length < 1.5 else "❌"
                print(f"  {status} {bone}: {length:.3f} (abs: {abs_length:.3f})")
                if not (0 < length < 1.5):
                    issues.append(f"{limb_type} {bone} length {length} out of range")

    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ All bone lengths valid")
        return True


def validate_material_properties():
    """Validate material property configurations"""
    print(f"\n{'='*70}")
    print("Validating Material Properties")
    print(f"{'='*70}\n")

    material_file = Path(__file__).parent.parent / "config" / "material" / "flexible_membrane.yaml"

    if not material_file.exists():
        print("❌ Material config not found")
        return False

    with open(material_file) as f:
        config = yaml.safe_load(f)

    regions = config.get('material_properties', {}).get('regions', {})

    issues = []

    for region_name, props in regions.items():
        print(f"\n{region_name}:")

        # Check density
        density = props.get('density', 0)
        status = "✓" if 0 < density < 10 else "❌"
        print(f"  {status} density: {density}")
        if not (0 < density < 10):
            issues.append(f"{region_name} density {density} out of range")

        # Check alpha values
        alpha_length = props.get('length_constraint_alpha', 0)
        status = "✓" if alpha_length > 0 else "❌"
        print(f"  {status} length_constraint_alpha: {alpha_length}")
        if alpha_length <= 0:
            issues.append(f"{region_name} length_constraint_alpha must be positive")

        alpha_bend = props.get('bend_constraint_alpha', 0)
        status = "✓" if alpha_bend > 0 else "❌"
        print(f"  {status} bend_constraint_alpha: {alpha_bend}")
        if alpha_bend <= 0:
            issues.append(f"{region_name} bend_constraint_alpha must be positive")

    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ All material properties valid")
        return True


def main():
    """Main debug function"""
    import argparse

    parser = argparse.ArgumentParser(description='Debug configuration files')
    parser.add_argument('--file', type=str, help='Specific YAML file to debug')
    parser.add_argument('--all', action='store_true', help='Check all configurations')

    args = parser.parse_args()

    if args.file:
        # Debug specific file
        debug_yaml_file(args.file)
    elif args.all:
        # Run all checks
        print("\n" + "="*70)
        print("CONFIGURATION DEBUG REPORT")
        print("="*70)

        checks = [
            ("Completeness", check_config_completeness()),
            ("Bone Lengths", validate_bone_lengths()),
            ("Material Properties", validate_material_properties()),
        ]

        # Debug each config file
        config_dir = Path(__file__).parent.parent / "config"
        config_files = [
            config_dir / "config.yaml",
            config_dir / "armature" / "base_rig.yaml",
            config_dir / "species" / "flying_squirrel_minimal.yaml",
            config_dir / "material" / "flexible_membrane.yaml",
            config_dir / "appendages" / "squirrel_standard.yaml",
        ]

        for cfg_file in config_files:
            if cfg_file.exists():
                result = debug_yaml_file(cfg_file)
                checks.append((cfg_file.name, result))

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        all_pass = True
        for name, result in checks:
            status = "✓ PASS" if result else "❌ FAIL"
            print(f"{status}: {name}")
            if not result:
                all_pass = False

        if all_pass:
            print("\n✓ All checks passed!")
            return 0
        else:
            print("\n❌ Some checks failed")
            return 1
    else:
        # Default: check completeness
        if check_config_completeness():
            print("\n✓ Configuration appears complete")
            print("Run with --all for detailed validation")
            return 0
        else:
            print("\n❌ Configuration incomplete")
            return 1


if __name__ == '__main__':
    sys.exit(main())
