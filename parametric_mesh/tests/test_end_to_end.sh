#!/bin/bash
# End-to-End Workflow Test
# Tests the complete pipeline from config to simulation

set -e  # Exit on error

echo "======================================================================"
echo "End-to-End Parametric Mesh Workflow Test"
echo "======================================================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test directory
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TEST_DIR/../.." && pwd)"
PARAM_DIR="$REPO_ROOT/parametric_mesh"

echo "Test directory: $TEST_DIR"
echo "Repository root: $REPO_ROOT"
echo ""

# Check Blender is available
if ! command -v blender &> /dev/null; then
    echo -e "${RED}✗ Blender not found${NC}"
    echo "  Please install Blender or add it to PATH"
    exit 1
fi

echo -e "${GREEN}✓ Blender found:${NC} $(blender --version | head -n1)"
echo ""

# Step 1: Run config tests
echo "======================================================================"
echo "Step 1: Testing Configuration Loading"
echo "======================================================================"
cd "$PARAM_DIR"
python tests/test_config_loading.py 2>&1 | grep -E "(✓|FAIL|ERROR|OK)" || true
echo ""

# Step 2: Generate mesh
echo "======================================================================"
echo "Step 2: Generating Test Mesh"
echo "======================================================================"
cd "$PARAM_DIR"

# Clean previous outputs
rm -rf generated/meshes/flying_squirrel_minimal.*
rm -rf generated/planforms/flying_squirrel_minimal_*

# Run mesh generator
echo "Running: blender --background --python scripts/mesh_generator.py"
blender --background --python scripts/mesh_generator.py 2>&1 | \
    grep -E "(Generator|Armature|Appendages|Body|Patagium|Regions|Validation|Export|Error)" || true

# Check outputs
echo ""
echo "Checking generated files..."

if [ -f "generated/meshes/flying_squirrel_minimal.blend" ]; then
    echo -e "${GREEN}✓ .blend file generated${NC}"
    ls -lh generated/meshes/flying_squirrel_minimal.blend
else
    echo -e "${RED}✗ .blend file NOT generated${NC}"
fi

if [ -f "generated/meshes/flying_squirrel_minimal.obj" ]; then
    echo -e "${GREEN}✓ .obj file generated${NC}"
    ls -lh generated/meshes/flying_squirrel_minimal.obj
else
    echo -e "${RED}✗ .obj file NOT generated${NC}"
fi

if [ -f "generated/meshes/flying_squirrel_minimal_metadata.json" ]; then
    echo -e "${GREEN}✓ metadata.json generated${NC}"
    ls -lh generated/meshes/flying_squirrel_minimal_metadata.json
else
    echo -e "${YELLOW}⚠ metadata.json NOT generated (optional)${NC}"
fi

echo ""

# Step 3: Validate mesh
echo "======================================================================"
echo "Step 3: Validating Generated Mesh"
echo "======================================================================"
cd "$PARAM_DIR"
python tests/test_mesh_validation.py 2>&1 | grep -E "(✓|⚠|FAIL|ERROR|OK|Validating)" || true
echo ""

# Step 4: Generate planiform
echo "======================================================================"
echo "Step 4: Generating SVG Planiform"
echo "======================================================================"
cd "$PARAM_DIR"

if [ -f "generated/meshes/flying_squirrel_minimal.blend" ]; then
    echo "Generating top view..."
    blender --background --python scripts/planiform_exporter.py -- \
        --blend-file generated/meshes/flying_squirrel_minimal.blend \
        --output generated/planforms/flying_squirrel_minimal_top.svg \
        --view top \
        --title "Flying Squirrel - Top View" 2>&1 | \
        grep -E "(Planiform|Error)" || true

    echo "Generating side view..."
    blender --background --python scripts/planiform_exporter.py -- \
        --blend-file generated/meshes/flying_squirrel_minimal.blend \
        --output generated/planforms/flying_squirrel_minimal_side.svg \
        --view side \
        --title "Flying Squirrel - Side View" 2>&1 | \
        grep -E "(Planiform|Error)" || true

    # Check SVG outputs
    if [ -f "generated/planforms/flying_squirrel_minimal_top.svg" ]; then
        echo -e "${GREEN}✓ Top view SVG generated${NC}"
    else
        echo -e "${RED}✗ Top view SVG NOT generated${NC}"
    fi

    if [ -f "generated/planforms/flying_squirrel_minimal_side.svg" ]; then
        echo -e "${GREEN}✓ Side view SVG generated${NC}"
    else
        echo -e "${RED}✗ Side view SVG NOT generated${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping planiform (no .blend file)${NC}"
fi

echo ""

# Step 5: Test simulation integration
echo "======================================================================"
echo "Step 5: Testing Simulation Integration"
echo "======================================================================"
cd "$PARAM_DIR"
python tests/test_simulation_integration.py 2>&1 | grep -E "(✓|FAIL|ERROR|OK)" || true
echo ""

# Step 6: Copy to simulation directory (optional)
echo "======================================================================"
echo "Step 6: Preparing for Simulation"
echo "======================================================================"

if [ -f "generated/meshes/flying_squirrel_minimal.obj" ]; then
    echo "Copying mesh to simulation directory..."
    cp generated/meshes/flying_squirrel_minimal.obj ../3D/assets/mesh/ || true

    if [ -f "generated/meshes/flying_squirrel_minimal_metadata.json" ]; then
        cp generated/meshes/flying_squirrel_minimal_metadata.json ../3D/assets/mesh/ || true
    fi

    echo -e "${GREEN}✓ Mesh files copied to 3D/assets/mesh/${NC}"
else
    echo -e "${YELLOW}⚠ No mesh to copy${NC}"
fi

echo ""

# Summary
echo "======================================================================"
echo "Test Summary"
echo "======================================================================"

FAIL_COUNT=0

# Check each expected output
if [ ! -f "generated/meshes/flying_squirrel_minimal.blend" ]; then
    echo -e "${RED}✗ Missing: .blend file${NC}"
    ((FAIL_COUNT++))
fi

if [ ! -f "generated/meshes/flying_squirrel_minimal.obj" ]; then
    echo -e "${RED}✗ Missing: .obj file${NC}"
    ((FAIL_COUNT++))
fi

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All critical outputs generated successfully${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. View mesh in Blender:"
    echo "     blender generated/meshes/flying_squirrel_minimal.blend"
    echo ""
    echo "  2. View planiform:"
    echo "     open generated/planforms/flying_squirrel_minimal_top.svg"
    echo ""
    echo "  3. Run simulation (if mesh looks good):"
    echo "     cd ../3D && python run.py"
    exit 0
else
    echo -e "${RED}✗ $FAIL_COUNT critical failure(s)${NC}"
    echo "Check the output above for errors"
    exit 1
fi
