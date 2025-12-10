# Project Assessment: PFM-SFI (Particle Flow Maps - Solid-Fluid Interaction)

## Overview

This is a SIGGRAPH ASIA 2024 research project implementing solid-fluid interaction simulations using particle flow maps. It simulates complex fluid dynamics with elastic solids, including swimming fish, flags, and parachutes with vortex interactions.

## Project Structure

```text
pfm-sfi/
├── 2D/              # 2D simulations (swimming fish example)
│   ├── run.py       # Main 2D simulator
│   ├── hyperparameters.py
│   ├── init_conditions.py
│   ├── mpm_utils.py (Material Point Method utilities)
│   ├── mgpcg_m.py   (Multigrid Preconditioned CG solver)
│   └── ...
├── 3D/              # 3D simulations (cloth, flags, parachutes)
│   ├── run.py       # Main 3D simulator
│   ├── hyperparameters.py
│   ├── init_conditions.py
│   ├── ibm_cloth.py (Immersed Boundary Method for cloth)
│   ├── assets/mesh/ (3D mesh assets)
│   └── utils/       (3D utilities, rendering, geometry)
└── README.md
```

## Key Technical Details

### Core Dependencies

- **Taichi** (main simulation framework with CUDA backend)
- **PyTorch** (3D version only)
- **NumPy** (likely implicit dependency)
- **Loguru** (for logging)
- Standard libraries: sys, shutil, time

### Hardware Requirements

- **NVIDIA GPU with CUDA support** (REQUIRED)
- 2D version: 3GB GPU memory minimum
- 3D version: 10.5GB GPU memory minimum

## Getting Started

### Step 1: Create Environment

Choose either **Option A (Conda/Mamba - Recommended)** or **Option B (Python venv)**:

#### Option A: Conda/Mamba Environment (Recommended)

Conda/Mamba automatically handles CUDA dependencies:

```bash
# Using conda
conda env create -f environment.yml
conda activate pfm-sfi

# OR using mamba (faster)
mamba env create -f environment.yml
mamba activate pfm-sfi
```

#### Option B: Python Virtual Environment

If you prefer venv, you'll need to ensure CUDA is installed separately:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

**Note:** With venv, you must have CUDA Toolkit installed separately on your system.

### Step 2: Verify CUDA Setup

```bash
# Check if Taichi can access CUDA
python -c "import taichi as ti; ti.init(arch=ti.cuda); print('CUDA available!')"
```

### Step 3: Run Simulations

**For 2D simulations (swimming fish):**

```bash
cd 2D
python run.py
```

**For 3D simulations (cloth/flags):**

```bash
cd 3D
python run.py
```

## Configuration

### 2D Configuration

File: `2D/hyperparameters.py:30-43`

- Resolution: 512×128
- Total frames: 1500
- CFL: 0.5
- Example: "2D_swim"

### 3D Configuration

File: `3D/hyperparameters.py:24-36`

- Resolution: 256×128×128
- Total frames: 1500
- CFL: 0.25
- Example: "3D_cloth4"

## Potential Issues & Solutions

1. **CUDA requirement** - Ensure you have an NVIDIA GPU with CUDA installed
   - Conda/Mamba: CUDA is included in environment.yml
   - venv: Install CUDA Toolkit separately from NVIDIA
2. **High memory usage** - Reduce resolution in hyperparameters.py if needed
3. **Output location** - Check `io_utils.py` files for output directory configuration
4. **PyTorch CUDA compatibility** - If you encounter CUDA version mismatches, update pytorch-cuda version in environment.yml to match your system

## Next Steps

1. Create environment using conda/mamba (recommended) or venv
2. Install dependencies (automatic with conda, or via requirements.txt with venv)
3. Verify CUDA is available
4. Choose 2D or 3D simulation
5. Modify `hyperparameters.py` to configure your simulation
6. Run the appropriate `run.py` file
7. Check output directory for results (frames/videos)

## Key Files to Review

- `2D/run.py:10` - CUDA initialization with 3GB memory allocation
- `3D/run.py:12` - CUDA initialization with 10.5GB memory allocation
- `2D/hyperparameters.py` - 2D simulation configuration
- `3D/hyperparameters.py` - 3D simulation configuration
- `2D/init_conditions.py` - Initial conditions for 2D scenarios
- `3D/init_conditions.py` - Initial conditions for 3D scenarios

## Reference Links

- Project Homepage: <https://cdwj.github.io/projects/pfm-sfi-project-page/index.html>
- Paper: [SIGGRAPH ASIA 2024 Paper](https://cdwj.github.io/projects/pfm-sfi-project-page/static/pdfs/SASIA_2024__Solid_Fluid_Interaction_on_Particle_Flow_Maps.pdf)
- Source Code: <https://github.com/CDWJ/pfm-sfi>
