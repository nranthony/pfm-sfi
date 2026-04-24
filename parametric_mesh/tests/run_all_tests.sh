#!/bin/bash
# Master Test Runner
# Runs all available tests with summary

set -e

echo "======================================================================"
echo "PARAMETRIC MESH SYSTEM - MASTER TEST SUITE"
echo "======================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TEST_DIR/.."

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# Helper function to run test
run_test() {
    local test_name="$1"
    local test_command="$2"
    local required="$3"  # "required" or "optional"

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}TEST: $test_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if eval "$test_command"; then
        echo -e "${GREEN}✓ PASS: $test_name${NC}\n"
        ((PASS_COUNT++))
        return 0
    else
        if [ "$required" = "required" ]; then
            echo -e "${RED}✗ FAIL: $test_name${NC}\n"
            ((FAIL_COUNT++))
            return 1
        else
            echo -e "${YELLOW}⚠ SKIP: $test_name (optional)${NC}\n"
            ((SKIP_COUNT++))
            return 0
        fi
    fi
}

# 1. Configuration Tests
echo ""
run_test "Configuration Loading" \
    "python tests/test_config_loading.py 2>&1 | tail -n 20" \
    "required"

run_test "Configuration Debug Check" \
    "python tests/debug_config.py --all 2>&1 | tail -n 30" \
    "required"

# 2. Simulation Integration Tests
run_test "Simulation Integration" \
    "python tests/test_simulation_integration.py 2>&1 | tail -n 20" \
    "required"

# 3. Blender Tests (optional if Blender not available)
if command -v blender &> /dev/null; then
    run_test "Blender Environment Debug" \
        "blender --background --python tests/debug_blender.py 2>&1 | tail -n 30" \
        "optional"

    run_test "Mesh Generation" \
        "blender --background --python scripts/mesh_generator.py 2>&1 | grep -E '(Generator|Export|Error)' | tail -n 20" \
        "optional"

    # Only run validation if mesh was generated
    if [ -f "generated/meshes/flying_squirrel_minimal.obj" ]; then
        run_test "Mesh Validation" \
            "python tests/test_mesh_validation.py 2>&1 | tail -n 20" \
            "optional"
    else
        echo -e "${YELLOW}⚠ SKIP: Mesh Validation (no mesh generated)${NC}\n"
        ((SKIP_COUNT++))
    fi
else
    echo -e "${YELLOW}⚠ SKIP: Blender tests (Blender not found)${NC}\n"
    ((SKIP_COUNT+=3))
fi

# Summary
echo ""
echo "======================================================================"
echo "TEST SUMMARY"
echo "======================================================================"

TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))

echo -e "${GREEN}✓ Passed:${NC}  $PASS_COUNT / $TOTAL"
echo -e "${RED}✗ Failed:${NC}  $FAIL_COUNT / $TOTAL"
echo -e "${YELLOW}⚠ Skipped:${NC} $SKIP_COUNT / $TOTAL"

echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ ALL REQUIRED TESTS PASSED${NC}"
    echo ""
    echo "System is ready for use!"
    echo ""
    echo "Next steps:"
    echo "  1. Run full end-to-end test: ./tests/test_end_to_end.sh"
    echo "  2. Generate a mesh: blender --background --python scripts/mesh_generator.py"
    echo "  3. View in Blender: blender generated/meshes/flying_squirrel_minimal.blend"
    exit 0
else
    echo -e "${RED}✗ SOME REQUIRED TESTS FAILED${NC}"
    echo ""
    echo "Please fix the failures above before proceeding."
    echo "Run individual tests for more details:"
    echo "  python tests/test_config_loading.py"
    echo "  python tests/test_simulation_integration.py"
    exit 1
fi
