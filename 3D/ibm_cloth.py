"""
IBM Cloth Configuration with YAML Support
Modified to support both legacy hardcoded configs and new YAML-based configs

Usage:
    # New YAML mode:
    cloth_config = create_cloth_config(config_path='configs/flying_squirrel_minimal.yaml')

    # Legacy mode (backward compatible):
    cloth_config = create_cloth_config(mesh_name='silk2.obj')
"""

import taichi as ti
import numpy as np
import os
from gmesh import *
from framework import *
from length import *
import meshio
from bend import *
from math import pi
from hyperparameters import *
from ibm_cloth_base import IBMClothConfig

# ============================================================================
# Configuration Selection
# ============================================================================

# Choose mode: 'yaml' or 'legacy'
MODE = os.getenv('IBM_CLOTH_MODE', 'legacy')  # Set via env var or change here

if MODE == 'yaml':
    # YAML mode: load from config file
    CONFIG_PATH = os.getenv('IBM_CLOTH_CONFIG', 'configs/flying_squirrel_minimal.yaml')
    print(f"[IBMCloth] Using YAML mode with config: {CONFIG_PATH}")
    cloth = IBMClothConfig(config_path=CONFIG_PATH)
    cloth.setup()

else:
    # Legacy mode: use hardcoded values (backward compatible)
    print("[IBMCloth] Using legacy mode with hardcoded config")
    cloth = IBMClothConfig(mesh_name='silk2.obj')
    cloth.setup()

# ============================================================================
# Export cloth components for use in simulation
# ============================================================================

# Mesh
mesh = cloth.mesh
ibm_dx = cloth.ibm_dx

# Constraints
xpbd = cloth.xpbd
length_cons = cloth.length_cons
bend_cons = cloth.bend_cons
solve_iters = cloth.get_solve_iters()
dt = cloth.get_dt()
g = cloth.get_gravity()

# Fixed points
cons_vert_i = cloth.cons_vert_i
cons_vert_p = cloth.cons_vert_p
cons_pos = cloth.cons_pos
cons_pos_init = cloth.cons_pos_init

# Force fields
pointForce = cloth.pointForce
pointForce_copy = cloth.pointForce_copy
pointLocation_copy = cloth.pointLocation_copy
mesh_vp_copy = cloth.mesh_vp_copy
mesh_vel_copy = cloth.mesh_vel_copy

# ============================================================================
# IBM Kernel and Functions (unchanged from original)
# ============================================================================

@ti.func
def ibm_kernel(dis):
    weight = 0.0
    r = ti.abs(dis)
    if ti.abs(dis) <= 2:
        weight = 0.25 * (1 + ti.cos(pi * 0.5 * r))
    return weight

def Export(path, i: int):
    npL = mesh.v_p.to_numpy()
    npI = mesh.f_i.to_numpy()
    import meshio
    out_mesh = meshio.Mesh(
        points=npL,
        cells=[('triangle', npI.reshape(-1, 3))]
    )
    meshio.write(f'{path}/solid_{i:04d}.ply', out_mesh, binary=True)

@ti.kernel
def copy_solid_velocity(u_x:ti.template(), u_y:ti.template(), u_z:ti.template()):
    for i in range(mesh.n_vert):
        vel = sample_ibm_u(u_x, u_y, u_z, mesh.v_p[i], ibm_dx)
        mesh_vel_copy[i] = vel

@ti.kernel
def update_force(u_x:ti.template(), u_y:ti.template(), u_z:ti.template(), dt:float):
    for i in range(mesh.n_vert):
        pointForce_copy[i] = pointForce[i]
        u_mesh = sample_ibm_u(u_x, u_y, u_z, mesh.v_p[i], ibm_dx)
        u_pbd = xpbd.v_v[i]
        pointForce[i] = (u_mesh - u_pbd) / dt
        pointLocation_copy[i] = mesh.v_p[i]
        mesh_vp_copy[i] = mesh.v_p[i]

@ti.kernel
def spread_force(u_x:ti.template(), u_y:ti.template(), u_z:ti.template(), dt:float):
    for i in range(mesh.n_vert):
        pos = mesh.v_p[i] / ibm_dx
        # x component
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 2))
        for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
                weight = ibm_kernel(pos[0] - face_id[0]) * ibm_kernel(pos[1] - face_id[1] - 0.5) * ibm_kernel(pos[2] - face_id[2] - 0.5)
                u_x[face_id] += pointForce[i][0] * weight * dt

        # y component
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 2))
        for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y and 0 <= face_id[2] < res_z:
                weight = ibm_kernel(pos[0] - face_id[0] - 0.5) * ibm_kernel(pos[1] - face_id[1]) * ibm_kernel(pos[2] - face_id[2] - 0.5)
                u_y[face_id] += pointForce[i][1] * weight * dt

        # z component
        base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 1))
        for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
            face_id = base_face_id + offset
            if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] <= res_z:
                weight = ibm_kernel(pos[0] - face_id[0] - 0.5) * ibm_kernel(pos[1] - face_id[1] - 0.5) * ibm_kernel(pos[2] - face_id[2])
                u_z[face_id] += pointForce[i][2] * weight * dt

@ti.func
def sample_ibm_u(u_x, u_y, u_z, p, dx):
    vel = ti.Vector([0.0, 0.0, 0.0])
    pos = p / dx

    # x component
    base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1) - 0.5 * ti.Vector.unit(dim, 2))
    for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
        face_id = base_face_id + offset
        if 0 <= face_id[0] <= res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] < res_z:
            weight = ibm_kernel(pos[0] - face_id[0]) * ibm_kernel(pos[1] - face_id[1] - 0.5) * ibm_kernel(pos[2] - face_id[2] - 0.5)
            vel[0] += u_x[face_id] * weight

    # y component
    base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 2))
    for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
        face_id = base_face_id + offset
        if 0 <= face_id[0] < res_x and 0 <= face_id[1] <= res_y and 0 <= face_id[2] < res_z:
            weight = ibm_kernel(pos[0] - face_id[0] - 0.5) * ibm_kernel(pos[1] - face_id[1]) * ibm_kernel(pos[2] - face_id[2] - 0.5)
            vel[1] += u_y[face_id] * weight

    # z component
    base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0) - 0.5 * ti.Vector.unit(dim, 1))
    for offset in ti.grouped(ti.ndrange(*((-3, 4),) * dim)):
        face_id = base_face_id + offset
        if 0 <= face_id[0] < res_x and 0 <= face_id[1] < res_y and 0 <= face_id[2] <= res_z:
            weight = ibm_kernel(pos[0] - face_id[0] - 0.5) * ibm_kernel(pos[1] - face_id[1] - 0.5) * ibm_kernel(pos[2] - face_id[2])
            vel[2] += u_z[face_id] * weight

    return vel

@ti.kernel
def advect_ibm(u_x:ti.template(), u_y:ti.template(), u_z:ti.template(), dt:float, change_p: float):
    for i in range(mesh.n_vert):
        u1 = sample_ibm_u(u_x, u_y, u_z, mesh.v_p[i], ibm_dx)
        psi_x1 = mesh.v_p[i] + 0.5 * dt * u1
        u2 = sample_ibm_u(u_x, u_y, u_z, psi_x1, ibm_dx)
        psi_x2 = mesh.v_p[i] + 0.5 * dt * u2
        u3 = sample_ibm_u(u_x, u_y, u_z, psi_x2, ibm_dx)
        psi_x3 = mesh.v_p[i] + 1.0 * dt * u3
        u4 = sample_ibm_u(u_x, u_y, u_z, psi_x3, ibm_dx)

        if change_p == 1:
            mesh.v_p[i] = mesh.v_p[i] + dt * 1./6 * (u1 + 2*u2 + 2*u3 + u4)

        xpbd.v_v[i] = sample_ibm_u(u_x, u_y, u_z, mesh.v_p[i], ibm_dx) + g*dt

def solve_for_xpbd(dt):
    cons_vert_p.from_numpy(cons_pos)
    mesh.set_pos_by_index(n=len(cons_pos), index=cons_vert_i, pos=cons_vert_p)
    length_cons.update_alpha(dt)
    xpbd.dt = dt
    bend_cons.update_alpha(dt)
    xpbd.preupdate_cons()
    for _ in range(solve_iters):
        xpbd.update_cons()
    xpbd.update_vel()

print("[IBMCloth] Module loaded successfully")
