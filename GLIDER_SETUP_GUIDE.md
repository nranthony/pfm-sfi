# Mammalian Glider Setup Guide

## Overview
This guide explains how to set up a deformable mammalian glider (flying squirrel, sugar glider, etc.) for fluid-structure interaction simulation using the PFM-SFI framework.

The glider uses:
- **IBM (Immersed Boundary Method)** for fluid coupling
- **XPBD (Extended Position Based Dynamics)** for elastic membrane
- **Blender-created mesh** with armature for skeletal structure

---

## Part 1: Creating the Glider Mesh in Blender

### 1.1 Basic Membrane Geometry

1. **Open Blender** and create a new project
2. **Delete default cube** (X → Delete)
3. **Add Plane** (Shift+A → Mesh → Plane)
4. **Subdivide** the plane for better resolution:
   - Tab (Edit Mode)
   - Right-click → Subdivide
   - Repeat 4-5 times for ~1000-2000 vertices
   - Or use Modifier: Add Subdivision Surface (Level 3-4)

5. **Shape the membrane**:
   - Select vertices in Edit Mode (Tab)
   - Use Proportional Editing (O key) to sculpt the patagium shape
   - Create a bat-wing or rectangular glider shape
   - Recommended dimensions (before scaling):
     - Width (X): 2.0 units
     - Length (Z): 1.5 units
     - Thickness: Leave as single-surface (membrane)

6. **Add slight curvature** (optional):
   - Select middle vertices
   - Move down slightly (G → Y → -0.1)
   - Creates natural droop/camber

### 1.2 Creating the Armature (Skeleton)

1. **Add Armature** (Shift+A → Armature → Single Bone)
2. **Build skeleton structure**:
   ```
   Recommended bone structure:
   - Spine (center, vertical)
   - FrontLeft_Limb (extends to front-left corner)
   - FrontRight_Limb (extends to front-right corner)
   - BackLeft_Limb (extends to back-left corner)
   - BackRight_Limb (extends to back-right corner)
   - [Optional] Head, Tail bones
   ```

3. **Position bones**:
   - Tab into Edit Mode for Armature
   - Position bone heads at body center
   - Extend bone tails to membrane corners
   - Align with mesh corner vertices

4. **Example armature layout** (top view):
   ```
        FrontLeft ●────────┐
                           │
                         ● Spine (center)
                           │
       BackLeft ●──────────┘

        FrontRight ●───────┐
                           │
                         ● Spine
                           │
       BackRight ●─────────┘
   ```

### 1.3 Skinning (Weight Painting)

1. **Select mesh**, then **Shift+Select armature**
2. **Parent with Automatic Weights**:
   - Ctrl+P → With Automatic Weights

3. **Refine weight painting**:
   - Select Armature, then mesh
   - Switch to Weight Paint Mode
   - Paint vertex weights for each bone:
     - Corner vertices → 100% weight to corresponding limb bone
     - Center vertices → weight to spine
     - Gradient weights in between

4. **Verify weights**:
   - Pose Mode (Ctrl+Tab)
   - Move limb bones
   - Check membrane deforms correctly

### 1.4 Export Settings

1. **Select the mesh** (not armature)
2. **File → Export → Wavefront (.obj)**
3. **Export settings**:
   - ☑ Selection Only
   - ☑ Apply Modifiers
   - ☑ Include Normals
   - ☑ Triangulate Faces (IMPORTANT!)
   - ☐ Include UVs (optional)
   - Forward: Y Forward
   - Up: Z Up
   - Scale: 1.00

4. **Save as**: `glider.obj` in `/3D/assets/mesh/`

5. **Export armature data** (for reference):
   - Select armature
   - Export as `glider_armature.json` (custom script) or note bone positions manually

### 1.5 Bone Position Reference

After export, note the bone tail positions in object coordinates. These will be used to define fixed points in the simulation.

Example bone positions (in normalized coordinates):
```python
# Front limbs (high Z)
front_left_bone  = (-1.0, 0.0, 0.8)   # X min, Z max
front_right_bone = ( 1.0, 0.0, 0.8)   # X max, Z max

# Back limbs (low Z)
back_left_bone   = (-1.0, 0.0, -0.8)  # X min, Z min
back_right_bone  = ( 1.0, 0.0, -0.8)  # X max, Z min

# Spine/center
spine_bone       = ( 0.0, 0.0, 0.0)   # Center
```

---

## Part 2: Python Implementation

### 2.1 Create `/3D/ibm_glider.py`

```python
import taichi as ti
import numpy as np
import os
from gmesh import *
from framework import *
from length import *
from bend import *
import meshio
from math import pi
from hyperparameters import *

# ============================================
# CONFIGURATION PARAMETERS
# ============================================

# Mesh file
MESH_FILE = 'glider.obj'

# Material properties
MEMBRANE_DENSITY = 0.5        # kg/m² (light mammalian skin)
MEMBRANE_SCALE = 0.4          # Size in simulation units
INITIAL_HEIGHT = 0.7          # Starting Y position (in air)

# Elastic properties
STRETCH_STIFFNESS = 0.008     # Lower = stiffer (alpha for length constraint)
BENDING_STIFFNESS = 2500      # Lower = more rigid (alpha for bend constraint)

# Physics
GRAVITY_STRENGTH = -9.8       # Standard earth gravity (m/s²)
TIME_STEP = 0.0005            # Simulation timestep
SOLVER_ITERATIONS = 50        # XPBD constraint solver iterations

# Bone positions (from Blender armature) - normalized coordinates
# These should match your Blender bone tail positions
BONE_POSITIONS = {
    'front_left':  np.array([-1.0, 0.0,  0.8]),
    'front_right': np.array([ 1.0, 0.0,  0.8]),
    'back_left':   np.array([-1.0, 0.0, -0.8]),
    'back_right':  np.array([ 1.0, 0.0, -0.8]),
    'spine':       np.array([ 0.0, 0.0,  0.0])
}

# ============================================
# MESH LOADING
# ============================================

ibm_dx = 1.0 / res_y

def obj_parser(filepath):
    """Parse OBJ file using meshio"""
    mesh = meshio.read(filepath)
    v, f = mesh.points, mesh.cells_dict['triangle']
    return v, f.flatten()

# Load glider mesh
filepath = os.path.join(os.getcwd(), 'assets', 'mesh', MESH_FILE)
verts, faces = obj_parser(filepath)

# Create mesh object
mesh = TrianMesh(
    verts, faces,
    dim=3,
    rho=MEMBRANE_DENSITY,
    scale=MEMBRANE_SCALE,
    repose=(0.5, INITIAL_HEIGHT, 0.5)  # Initial position (x, y, z)
)

print(f"Glider mesh loaded: {mesh.n_vert} vertices, {mesh.n_face} faces")

# ============================================
# XPBD FRAMEWORK SETUP
# ============================================

# Gravity vector
g = ti.Vector([0.0, GRAVITY_STRENGTH, 0.0])

# Create XPBD solver
xpbd = pbd_framework(
    g=g,
    n_vert=mesh.n_vert,
    v_p=mesh.v_p,
    dt=TIME_STEP
)

# Length constraint (prevents stretching)
length_cons = LengthCons(
    mesh.v_p,
    mesh.v_p_ref,
    mesh.e_i,
    mesh.v_invm,
    dt=TIME_STEP,
    alpha=STRETCH_STIFFNESS  # Lower = stiffer membrane
)

# Bending constraint (controls membrane flexibility)
bend_cons = Bend3D(
    mesh.v_p,
    mesh.v_p_ref,
    mesh.e_i,
    mesh.e_sidei,
    mesh.v_invm,
    dt=TIME_STEP,
    alpha=BENDING_STIFFNESS  # Lower = more rigid
)

# Add constraints to solver
xpbd.add_cons(length_cons)
xpbd.add_cons(bend_cons)
xpbd.init_rest_status()

# ============================================
# SKELETON/LIMB FIXED POINTS
# ============================================

def find_vertices_near_bone(mesh_vertices, bone_position, radius=0.15):
    """
    Find mesh vertices near a bone position (limb attachment points)

    Args:
        mesh_vertices: numpy array of vertex positions
        bone_position: 3D position of bone (normalized)
        radius: search radius around bone position

    Returns:
        indices of vertices within radius of bone
    """
    # Transform bone position to mesh scale
    bone_pos_scaled = bone_position * MEMBRANE_SCALE
    bone_pos_scaled[0] += 0.5  # Offset for repose
    bone_pos_scaled[1] += INITIAL_HEIGHT
    bone_pos_scaled[2] += 0.5

    # Calculate distances
    distances = np.linalg.norm(mesh_vertices - bone_pos_scaled, axis=1)

    # Find vertices within radius
    indices = np.where(distances < radius)[0]

    return indices

# Get mesh vertices
vertices_np = mesh.v_p.to_numpy()

# Find vertices attached to each limb bone
limb_attachments = {}
all_limb_indices = []

for bone_name, bone_pos in BONE_POSITIONS.items():
    if bone_name != 'spine':  # Don't fix spine vertices
        indices = find_vertices_near_bone(vertices_np, bone_pos, radius=0.1)
        limb_attachments[bone_name] = indices
        all_limb_indices.extend(indices)
        print(f"Bone '{bone_name}': {len(indices)} vertices attached")

# Convert to numpy array and remove duplicates
limb_indices = np.unique(np.array(all_limb_indices, dtype=np.int32))

print(f"Total limb attachment vertices: {len(limb_indices)}")

# Create Taichi fields for limb constraints
cons_vert_i = ti.field(dtype=ti.i32, shape=len(limb_indices))
cons_vert_i.from_numpy(limb_indices)

cons_vert_p = ti.Vector.field(3, dtype=ti.f32, shape=len(limb_indices))
mesh.get_pos_by_index(n=len(limb_indices), index=cons_vert_i, pos=cons_vert_p)

# Set limb vertices as fixed (infinite mass)
mesh.set_fixed_point(n=len(limb_indices), index=cons_vert_i)

# Save initial limb positions
cons_pos = cons_vert_p.to_numpy()
cons_pos_init = np.copy(cons_pos)

# ============================================
# LIMB CONTROL (OPTIONAL)
# ============================================

# Allow dynamic limb position control (for gliding maneuvers)
limb_control_enabled = ti.field(dtype=ti.i32, shape=())
limb_control_enabled[None] = 0  # 0=disabled, 1=enabled

@ti.kernel
def update_limb_positions(time: ti.f32, control_amplitude: ti.f32):
    """
    Dynamically update limb positions (e.g., for steering, wing adjustment)

    Args:
        time: simulation time
        control_amplitude: strength of limb movement
    """
    if limb_control_enabled[None]:
        # Example: Oscillate limbs to simulate gliding adjustments
        angle = ti.sin(time * 2.0) * control_amplitude

        for i in range(cons_vert_i.shape[0]):
            idx = cons_vert_i[i]
            # Move limbs up/down slightly
            cons_vert_p[i][1] = cons_pos_init[i][1] + angle

@ti.kernel
def apply_limb_constraints():
    """Apply position constraints from limb bones to mesh"""
    for i in range(cons_vert_i.shape[0]):
        idx = cons_vert_i[i]
        mesh.v_p[idx] = cons_vert_p[i]

# ============================================
# INITIAL SHAPE/CURVATURE
# ============================================

@ti.kernel
def add_initial_curvature(curvature_amount: ti.f32):
    """
    Add initial curved shape to membrane (camber/droop)
    Creates more realistic gliding aerodynamics

    Args:
        curvature_amount: strength of curvature (0.0-0.1 typical)
    """
    for i in range(mesh.n_vert):
        # Calculate distance from edges
        x_center = 0.5
        z_center = 0.5

        # Normalized distance from center (0 at edge, 1 at center)
        dist_x = 1.0 - ti.abs(mesh.v_p[i][0] - x_center) / (MEMBRANE_SCALE * 0.5)
        dist_z = 1.0 - ti.abs(mesh.v_p[i][2] - z_center) / (MEMBRANE_SCALE * 0.5)

        # Combined curvature factor (parabolic)
        curve_factor = dist_x * dist_z

        # Apply downward curvature
        mesh.v_p[i][1] -= curve_factor * curvature_amount

# Apply initial curvature (optional - comment out if not needed)
add_initial_curvature(0.03)  # 3cm droop at center

# ============================================
# FLUID COUPLING FIELDS
# ============================================

# Force buffers for fluid-solid coupling
pointForce = ti.Vector.field(3, dtype=ti.f32, shape=mesh.n_vert)
pointForce_copy = ti.Vector.field(3, dtype=ti.f32, shape=mesh.n_vert)
pointLocation_copy = ti.Vector.field(3, dtype=ti.f32, shape=mesh.n_vert)

mesh_vp_copy = ti.Vector.field(3, dtype=ti.f32, shape=mesh.n_vert)
mesh_vel_copy = ti.Vector.field(3, dtype=ti.f32, shape=mesh.n_vert)

# ============================================
# IBM KERNEL FUNCTIONS
# ============================================

@ti.func
def ibm_kernel(dis):
    """
    Immersed boundary kernel function
    Smoothed delta function for force spreading/interpolation
    """
    weight = 0.0
    r = ti.abs(dis)
    if r <= 2.0:
        weight = 0.25 * (1.0 + ti.cos(pi * 0.5 * r))
    return weight

# ============================================
# FORCE SPREADING (SOLID → FLUID)
# ============================================

@ti.kernel
def spread_force(u_x: ti.template(), u_y: ti.template(), u_z: ti.template(), dt: ti.f32):
    """
    Spread forces from solid vertices to fluid grid
    This is how the glider affects the fluid flow
    """
    for i in range(mesh.n_vert):
        pos = mesh.v_p[i] / ibm_dx

        # X-component (horizontal)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 2))
        for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
                weight = (ibm_kernel(pos[0] - face_id[0]) *
                         ibm_kernel(pos[1] - face_id[1] - 0.5) *
                         ibm_kernel(pos[2] - face_id[2] - 0.5))
                u_x[face_id] += pointForce[i][0] * weight * dt

        # Y-component (vertical)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 2))
        for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y and 0 <= face_id[2] < res_z:
                weight = (ibm_kernel(pos[0] - face_id[0] - 0.5) *
                         ibm_kernel(pos[1] - face_id[1]) *
                         ibm_kernel(pos[2] - face_id[2] - 0.5))
                u_y[face_id] += pointForce[i][1] * weight * dt

        # Z-component (depth)
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] <= res_z:
                weight = (ibm_kernel(pos[0] - face_id[0] - 0.5) *
                         ibm_kernel(pos[1] - face_id[1] - 0.5) *
                         ibm_kernel(pos[2] - face_id[2]))
                u_z[face_id] += pointForce[i][2] * weight * dt

@ti.kernel
def spread_rho(rho: ti.template(), dt: ti.f32):
    """Spread density from solid to fluid grid (for visualization)"""
    for i in range(mesh.n_vert):
        pos = mesh.v_p[i] / ibm_dx

        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) -
                          0.5 * ti.Vector.unit(dim, 1) -
                          0.5 * ti.Vector.unit(dim, 2))

        for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
            face_id = base_face_id + offset
            if (0 <= face_id[0] < res_x and
                0 <= face_id[1] < res_y and
                0 <= face_id[2] < res_z):
                weight = (ibm_kernel(pos[0] - face_id[0] - 0.5) *
                         ibm_kernel(pos[1] - face_id[1] - 0.5) *
                         ibm_kernel(pos[2] - face_id[2] - 0.5))
                rho[face_id] += MEMBRANE_DENSITY * weight

# ============================================
# VELOCITY INTERPOLATION (FLUID → SOLID)
# ============================================

@ti.func
def sample_ibm_u(u_x, u_y, u_z, p, dx):
    """
    Sample fluid velocity at a point using IBM kernel interpolation
    This is how the fluid affects the glider motion
    """
    vel = ti.Vector([0.0, 0.0, 0.0])
    pos = p / dx

    # Sample X velocity
    base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 2))
    for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
        face_id = base_face_id + offset
        if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
            weight = (ibm_kernel(pos[0] - face_id[0]) *
                     ibm_kernel(pos[1] - face_id[1] - 0.5) *
                     ibm_kernel(pos[2] - face_id[2] - 0.5))
            vel[0] += u_x[face_id] * weight

    # Sample Y velocity
    base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 2))
    for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
        face_id = base_face_id + offset
        if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y and 0 <= face_id[2] < res_z:
            weight = (ibm_kernel(pos[0] - face_id[0] - 0.5) *
                     ibm_kernel(pos[1] - face_id[1]) *
                     ibm_kernel(pos[2] - face_id[2] - 0.5))
            vel[1] += u_y[face_id] * weight

    # Sample Z velocity
    base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 1))
    for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
        face_id = base_face_id + offset
        if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] <= res_z:
            weight = (ibm_kernel(pos[0] - face_id[0] - 0.5) *
                     ibm_kernel(pos[1] - face_id[1] - 0.5) *
                     ibm_kernel(pos[2] - face_id[2]))
            vel[2] += u_z[face_id] * weight

    return vel

# ============================================
# ADVECTION (RK4 TIME INTEGRATION)
# ============================================

@ti.kernel
def advect_ibm(u_x: ti.template(), u_y: ti.template(), u_z: ti.template(),
               dt: ti.f32, change_p: ti.f32):
    """
    Advect glider vertices using RK4 integration
    change_p: if 1, update positions; if 0, only update velocities
    """
    for i in range(mesh.n_vert):
        # RK4 stage 1
        u1 = sample_ibm_u(u_x, u_y, u_z, mesh.v_p[i], ibm_dx)
        psi_x1 = mesh.v_p[i] + 0.5 * dt * u1

        # RK4 stage 2
        u2 = sample_ibm_u(u_x, u_y, u_z, psi_x1, ibm_dx)
        psi_x2 = mesh.v_p[i] + 0.5 * dt * u2

        # RK4 stage 3
        u3 = sample_ibm_u(u_x, u_y, u_z, psi_x2, ibm_dx)
        psi_x3 = mesh.v_p[i] + 1.0 * dt * u3

        # RK4 stage 4
        u4 = sample_ibm_u(u_x, u_y, u_z, psi_x3, ibm_dx)

        # Update position if requested
        if change_p == 1.0:
            mesh.v_p[i] = mesh.v_p[i] + dt * (u1 + 2.0*u2 + 2.0*u3 + u4) / 6.0

        # Update velocity for XPBD
        xpbd.v_v[i] = sample_ibm_u(u_x, u_y, u_z, mesh.v_p[i], ibm_dx) + g * dt

# ============================================
# VISUALIZATION/EXPORT
# ============================================

def Export(path, i: int):
    """Export glider mesh to PLY file for visualization"""
    npL = mesh.v_p.to_numpy()
    npI = mesh.f_i.to_numpy()

    mesh_writer = ti.tools.PLYWriter(
        num_vertices=mesh.n_vert,
        num_faces=mesh.n_face,
        face_type="tri"
    )

    # Scale back to visualization coordinates
    mesh_writer.add_vertex_pos(
        npL[:, 0] * res_y - 0.5,
        npL[:, 1] * res_y - 0.5,
        npL[:, 2] * res_y - 0.5
    )
    mesh_writer.add_faces(npI)
    mesh_writer.export_frame_ascii(i, f'{path}/glider_{i:06d}.ply')

# ============================================
# USAGE EXAMPLE (in main simulation loop)
# ============================================
"""
# Import this module in your main simulation file

from ibm_glider import *

# In your simulation loop:
for step in range(total_steps):

    # 1. Advect glider with fluid
    advect_ibm(u_x, u_y, u_z, dt, change_p=1)

    # 2. Update limb positions (optional - for active control)
    update_limb_positions(time=step*dt, control_amplitude=0.05)
    apply_limb_constraints()

    # 3. Solve XPBD constraints (elastic deformation)
    num_substeps = 5
    for substep in range(num_substeps):
        xpbd.make_prediction()
        for iteration in range(SOLVER_ITERATIONS):
            xpbd.update_cons()
        xpbd.update_vel()

    # 4. Compute coupling force (velocity difference)
    compute_coupling_force(pointForce, mesh_vp_copy, mesh.v_p, dt)

    # 5. Spread force to fluid grid
    spread_force(u_x, u_y, u_z, dt)

    # 6. Export for visualization
    if step % 10 == 0:
        Export('./output', step)
"""
```

---

## Part 3: Material Property Tuning Guide

### 3.1 Membrane Properties

| Property | Parameter | Realistic Range | Effect on Gliding |
|----------|-----------|-----------------|-------------------|
| **Density** | `MEMBRANE_DENSITY` | 0.3 - 0.8 kg/m² | Lower = floats more, higher = faster descent |
| **Stretch Stiffness** | `STRETCH_STIFFNESS` | 0.005 - 0.02 | Lower = tighter membrane, less flapping |
| **Bending Stiffness** | `BENDING_STIFFNESS` | 1000 - 5000 | Lower = holds shape better, higher = more billowing |

### 3.2 Tuning for Different Gliders

**Flying Squirrel** (flexible, billowing membrane):
```python
MEMBRANE_DENSITY = 0.4
STRETCH_STIFFNESS = 0.012
BENDING_STIFFNESS = 3500
```

**Sugar Glider** (smaller, lighter):
```python
MEMBRANE_DENSITY = 0.3
STRETCH_STIFFNESS = 0.008
BENDING_STIFFNESS = 2000
```

**Colugo/Flying Lemur** (larger, stiffer):
```python
MEMBRANE_DENSITY = 0.6
STRETCH_STIFFNESS = 0.006
BENDING_STIFFNESS = 1500
```

### 3.3 Debugging Tips

**Problem**: Membrane too floppy/unstable
- **Solution**: Decrease `BENDING_STIFFNESS` (make more rigid)
- Or decrease `STRETCH_STIFFNESS` (make tighter)

**Problem**: Glider falls too fast
- **Solution**: Decrease `MEMBRANE_DENSITY`
- Or increase `MEMBRANE_SCALE` (larger surface area)

**Problem**: Membrane tears/explodes
- **Solution**: Decrease `TIME_STEP` (smaller dt)
- Or increase `SOLVER_ITERATIONS`
- Or check mesh has no duplicate vertices

---

## Part 4: Integration with Main Simulation

### 4.1 Modify `3D/run.py` to include glider

Add to imports:
```python
from ibm_glider import (
    mesh, xpbd, pointForce,
    advect_ibm, spread_force, spread_rho,
    update_limb_positions, apply_limb_constraints,
    Export as ExportGlider,
    SOLVER_ITERATIONS, TIME_STEP
)
```

Add to simulation loop (around line 200+):
```python
# After fluid advection, before Poisson solve:

# 1. Advect glider
advect_ibm(u_x, u_y, u_z, dt, change_p=1)

# 2. Optional: Control limbs
if enable_limb_control:
    update_limb_positions(time=frame*dt, control_amplitude=0.05)
    apply_limb_constraints()

# 3. XPBD substeps
num_solid_substeps = 5
for substep in range(num_solid_substeps):
    xpbd.make_prediction()
    for iteration in range(SOLVER_ITERATIONS):
        xpbd.update_cons()
    xpbd.update_vel()

# 4. Compute and spread force
# (Add compute_coupling_force function similar to ibm_cloth.py)
spread_force(u_x, u_y, u_z, dt)

# 5. Continue with Poisson solve...
```

### 4.2 Visualization Setup

Render glider in Taichi GUI:
```python
window = ti.ui.Window("Glider Simulation", (1024, 1024))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

while window.running:
    # Render glider mesh
    scene.mesh(
        mesh.v_p,
        mesh.f_i,
        color=(0.8, 0.6, 0.4),  # Brownish (mammal fur)
        show_wireframe=False
    )

    # Render fluid particles
    scene.particles(...)

    canvas.scene(scene)
    window.show()
```

---

## Part 5: Advanced Features

### 5.1 Active Limb Control (Steering)

Implement dynamic limb movement for gliding control:

```python
# In ibm_glider.py, add:

@ti.kernel
def control_limb_asymmetric(time: ti.f32, left_amplitude: ti.f32, right_amplitude: ti.f32):
    """
    Asymmetric limb control for turning/steering
    """
    for i in range(cons_vert_i.shape[0]):
        idx = cons_vert_i[i]
        original_pos = cons_pos_init[i]

        # Determine if this vertex is on left or right side
        if original_pos[0] < 0.5:  # Left side
            angle = ti.sin(time * 2.0) * left_amplitude
        else:  # Right side
            angle = ti.sin(time * 2.0) * right_amplitude

        cons_vert_p[i][1] = original_pos[1] + angle
```

Usage:
```python
# Turn left: raise left limbs, lower right limbs
control_limb_asymmetric(time, left_amplitude=0.1, right_amplitude=-0.05)

# Turn right: opposite
control_limb_asymmetric(time, left_amplitude=-0.05, right_amplitude=0.1)
```

### 5.2 Adding Tail Control

If your mesh includes a tail:

```python
# Add to BONE_POSITIONS:
'tail': np.array([0.0, 0.0, -1.2])

# Control tail angle for pitch control
@ti.kernel
def control_tail(angle: ti.f32):
    """Adjust tail angle for pitch control"""
    # Find tail vertices and rotate them
    # ... implementation details
```

### 5.3 Muscle Activation (Active Strain)

For more realistic muscle contraction:

```python
@ti.kernel
def apply_muscle_activation(activation_region: ti.f32, strength: ti.f32):
    """
    Apply active strain deformation (similar to fish swimming)
    """
    for i in range(mesh.n_vert):
        # Check if vertex is in activation region
        if mesh.v_p[i][2] > activation_region:
            # Apply contraction using deformation gradient
            # F = Ftotal * Fa^-1 (see paper Eq. 25-27)
            pass
```

---

## Part 6: Validation & Testing

### 6.1 Test Checklist

- [ ] Mesh loads without errors
- [ ] Limb vertices are correctly identified
- [ ] Glider doesn't explode in first frame
- [ ] Membrane shows visible deformation
- [ ] Vortices form behind glider
- [ ] Glider descends at reasonable rate (not falling like a rock)
- [ ] Membrane billows/flutters in flow
- [ ] Export generates valid PLY files

### 6.2 Expected Behavior

**Good signs:**
- Smooth descent with gentle swaying
- Membrane cups/billows in airflow
- Visible vortex shedding at trailing edge
- Gradual deceleration as lift develops

**Bad signs:**
- Immediate collapse/folding
- Spinning/tumbling motion
- No visible deformation
- Extremely fast descent

### 6.3 Performance Benchmarks

Expected simulation performance (NVIDIA RTX 3080):
- **Grid resolution**: 256×128×128
- **Glider vertices**: 1000-2000
- **Timestep**: ~1.5-3 seconds per frame
- **Memory**: 4-6 GB GPU RAM

---

## Part 7: Troubleshooting

### Common Issues

**Issue**: Mesh not loading
```
Error: KeyError: 'triangle'
```
**Fix**: Ensure mesh is triangulated in Blender export settings

---

**Issue**: Glider explodes/NaN values
```
Error: nan detected in velocity
```
**Fix**:
1. Reduce `TIME_STEP` to 0.0001
2. Increase `SOLVER_ITERATIONS` to 100
3. Check for duplicate vertices in mesh

---

**Issue**: No deformation visible
**Fix**:
1. Check `STRETCH_STIFFNESS` and `BENDING_STIFFNESS` are not too low
2. Verify constraints are added: `xpbd.cons_list` should not be empty
3. Ensure limb vertices are actually fixed (check `mesh.v_invm` values)

---

**Issue**: Glider falls straight down (no gliding)
**Fix**:
1. Decrease `MEMBRANE_DENSITY` (too heavy)
2. Increase `MEMBRANE_SCALE` (needs more surface area)
3. Add initial velocity or angle
4. Check airflow direction matches glider orientation

---

**Issue**: Limbs not staying fixed
**Fix**:
1. Verify `find_vertices_near_bone()` returns vertices
2. Check bone positions match mesh geometry
3. Ensure `mesh.set_fixed_point()` is called before simulation

---

## Part 8: Example Configurations

### 8.1 Minimal Working Example

Save as `test_glider_simple.py`:

```python
import taichi as ti
ti.init(arch=ti.cuda)

# Import glider setup
from ibm_glider import *

# Simple test: Drop glider in still air
u_x = ti.field(dtype=ti.f32, shape=(res_x+1, res_y, res_z))
u_y = ti.field(dtype=ti.f32, shape=(res_x, res_y+1, res_z))
u_z = ti.field(dtype=ti.f32, shape=(res_x, res_y, res_z+1))

# Fill with zero velocity (still air)
u_x.fill(0.0)
u_y.fill(0.0)
u_z.fill(0.0)

# Run for 100 steps
for step in range(100):
    advect_ibm(u_x, u_y, u_z, TIME_STEP, change_p=1)

    xpbd.make_prediction()
    for i in range(SOLVER_ITERATIONS):
        xpbd.update_cons()
    xpbd.update_vel()

    if step % 10 == 0:
        print(f"Step {step}, Y position: {mesh.v_p.to_numpy()[0, 1]:.4f}")
        Export('./test_output', step)

print("Test complete!")
```

Run: `python test_glider_simple.py`

Expected output: Glider should descend smoothly, membrane should sag slightly

---

## Appendix A: Blender Python Script for Bone Position Export

Save as `export_bone_positions.py` in Blender:

```python
import bpy
import json

armature = bpy.data.objects['Armature']  # Change name if needed
bone_positions = {}

for bone in armature.pose.bones:
    # Get world space tail position
    tail_world = armature.matrix_world @ bone.tail
    bone_positions[bone.name] = list(tail_world)

# Save to JSON
output_file = '/path/to/your/project/glider_bones.json'
with open(output_file, 'w') as f:
    json.dump(bone_positions, f, indent=2)

print(f"Bone positions saved to {output_file}")
```

Run in Blender's Scripting tab, then load in Python:
```python
import json
with open('glider_bones.json', 'r') as f:
    BONE_POSITIONS = json.load(f)
```

---

## Appendix B: References

- **Paper**: "Solid-Fluid Interaction on Particle Flow Maps" (SASIA 2024)
  - Section 6.2: IBM Coupling (page 8)
  - Section 8.4: Active Strain (page 11)

- **XPBD Reference**: "A Survey on Position Based Simulation Methods" (Bender et al. 2014)

- **Mammalian Gliding Biomechanics**:
  - Byrnes et al. (2008) "Take-off and landing forces of flying squirrels"
  - Bishop (2006) "The relationship between 3-D kinematics and gliding performance"

---

## Appendix C: Parameter Quick Reference

```python
# Copy-paste configurations for different scenarios

# REALISTIC FLYING SQUIRREL
MEMBRANE_DENSITY = 0.4
STRETCH_STIFFNESS = 0.012
BENDING_STIFFNESS = 3500
MEMBRANE_SCALE = 0.4
GRAVITY_STRENGTH = -9.8

# HIGH PERFORMANCE (faster simulation)
SOLVER_ITERATIONS = 30
TIME_STEP = 0.001

# HIGH QUALITY (more accurate)
SOLVER_ITERATIONS = 100
TIME_STEP = 0.0002

# DEBUGGING (stable but slow)
SOLVER_ITERATIONS = 150
TIME_STEP = 0.0001
```

---

**End of Guide**

For questions or issues, refer to the main simulation documentation or the paper appendices.
