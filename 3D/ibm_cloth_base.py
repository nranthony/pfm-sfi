"""
Base class for IBM Cloth Configuration
Loads mesh and parameters from YAML config files
"""

import taichi as ti
import numpy as np
import os
import yaml
import json
from pathlib import Path
from gmesh import *
from framework import *
from length import *
import meshio
from bend import *
from math import pi
from hyperparameters import *


class IBMClothConfig:
    """Base configuration class for IBM cloth simulation"""

    def __init__(self, config_path: str = None, mesh_name: str = None):
        """
        Initialize IBM cloth configuration

        Args:
            config_path: Path to YAML config file (optional)
            mesh_name: Name of mesh file (if not using config)
        """
        self.config = None
        self.mesh_path = None
        self.metadata_path = None
        self.metadata = None

        if config_path:
            self.load_config(config_path)
        elif mesh_name:
            # Legacy mode: use mesh name directly
            self.mesh_path = os.path.join(os.getcwd(), 'assets', 'mesh', mesh_name)
            self.use_defaults()
        else:
            raise ValueError("Must provide either config_path or mesh_name")

        # IBM parameters
        self.ibm_dx = 1.0 / res_y

        # Initialize mesh and constraints
        self.mesh = None
        self.xpbd = None
        self.length_cons = None
        self.bend_cons = None

        # Fixed constraints
        self.cons_vert_i = None
        self.cons_vert_p = None
        self.cons_pos = None
        self.cons_pos_init = None

        # Force fields
        self.pointForce = None
        self.pointForce_copy = None
        self.pointLocation_copy = None
        self.mesh_vp_copy = None
        self.mesh_vel_copy = None

    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print(f"[IBMCloth] Loaded config: {config_path}")

        # Extract mesh path from config
        mesh_name = self.config['metadata']['name'] + '.obj'
        self.mesh_path = os.path.join(os.getcwd(), 'assets', 'mesh', mesh_name)

        # Look for metadata file
        metadata_name = self.config['metadata']['name'] + '_metadata.json'
        metadata_path = Path(os.getcwd()) / 'assets' / 'mesh' / metadata_name
        if metadata_path.exists():
            self.metadata_path = metadata_path
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

    def use_defaults(self):
        """Use default parameters (for legacy mode)"""
        self.config = {
            'simulation': {
                'solve_iters': 50,
                'dt': 0.0005,
                'gravity': [0.0, 5.0, -3.0]
            },
            'material_properties': {
                'global_scale': 0.6,
                'repose_position': [0.2, 0.5, 0.35],
                'regions': {
                    'membrane': {
                        'density': 2.0,
                        'length_constraint_alpha': 0.015,
                        'bend_constraint_alpha': 6000
                    }
                }
            }
        }

    def obj_parser(self, filepath):
        """Parse OBJ file"""
        mesh = meshio.read(filepath)
        v, f = mesh.points, mesh.cells_dict['triangle']
        return v, f.flatten()

    def initialize_mesh(self):
        """Load and initialize mesh"""
        print(f"[IBMCloth] Loading mesh: {self.mesh_path}")

        verts, faces = self.obj_parser(self.mesh_path)

        # Get material properties
        mat_props = self.config['material_properties']['regions']['membrane']
        rho = mat_props.get('density', 2.0)
        scale = self.config['material_properties']['global_scale']
        repose = self.config['material_properties']['repose_position']

        # Create mesh
        self.mesh = TrianMesh(
            verts, faces,
            dim=3,
            rho=rho,
            scale=scale,
            repose=tuple(repose)
        )

        print(f"[IBMCloth] Mesh initialized: {self.mesh.n_vert} vertices, {self.mesh.n_face} faces")

    def initialize_constraints(self):
        """Initialize XPBD constraints"""
        sim_config = self.config['simulation']
        mat_props = self.config['material_properties']['regions']['membrane']

        # Simulation parameters
        solve_iters = sim_config.get('solve_iters', 50)
        dt = sim_config.get('dt', 0.0005)
        gravity = sim_config.get('gravity', [0.0, 5.0, -3.0])

        # Create gravity vector
        g = ti.Vector(gravity)

        # Create XPBD framework
        self.xpbd = pbd_framework(g=g, n_vert=self.mesh.n_vert, v_p=self.mesh.v_p, dt=dt)

        # Length constraints
        alpha_length = mat_props.get('length_constraint_alpha', 0.015)
        self.length_cons = LengthCons(
            self.mesh.v_p,
            self.mesh.v_p_ref,
            self.mesh.e_i,
            self.mesh.v_invm,
            dt=dt,
            alpha=alpha_length
        )

        # Bend constraints
        alpha_bend = mat_props.get('bend_constraint_alpha', 6000)
        self.bend_cons = Bend3D(
            self.mesh.v_p,
            self.mesh.v_p_ref,
            self.mesh.e_i,
            self.mesh.e_sidei,
            self.mesh.v_invm,
            dt=dt,
            alpha=alpha_bend
        )

        self.xpbd.add_cons(self.length_cons)
        self.xpbd.add_cons(self.bend_cons)
        self.xpbd.init_rest_status()

        print(f"[IBMCloth] Initialized constraints: alpha_length={alpha_length}, alpha_bend={alpha_bend}")

    def initialize_fixed_points(self):
        """Initialize fixed point constraints.

        Selection precedence:
          1. config.constraints.fixed_region — dispatched on `type` (bbox | indices)
          2. metadata.json fixed_indices — explicit list written by mesh generator
          3. legacy spatial rule (x < 0.205) — silk-flag compatibility
        """
        region = (self.config.get('constraints') or {}).get('fixed_region')
        if region is not None:
            indices = self._fixed_indices_from_region(region)
        elif self.metadata and 'fixed_indices' in self.metadata:
            indices = np.asarray(self.metadata['fixed_indices'], dtype=np.int32)
        else:
            indices = np.where(self.mesh.v_p.to_numpy()[:, 0] < 0.205)[0]

        if len(indices) == 0:
            print("[IBMCloth] Warning: No fixed points defined")
            return

        indices = indices.astype(np.int32)

        self.cons_vert_i = ti.field(dtype=ti.i32, shape=indices.shape[0])
        self.cons_vert_i.from_numpy(indices)

        self.cons_vert_p = ti.Vector.field(3, dtype=ti.f32, shape=indices.shape[0])
        self.mesh.get_pos_by_index(n=indices.shape[0], index=self.cons_vert_i, pos=self.cons_vert_p)
        self.mesh.set_fixed_point(n=indices.shape[0], index=self.cons_vert_i)

        self.cons_pos = self.cons_vert_p.to_numpy()
        self.cons_pos_init = np.copy(self.cons_pos)

        print(f"[IBMCloth] Fixed {len(indices)} vertices")

    def _fixed_indices_from_region(self, region: dict) -> np.ndarray:
        """Resolve a YAML `fixed_region` spec to vertex indices.

        Coordinates are in post-normalization domain space (same frame as
        repose_position), since `mesh.v_p` has already been scaled and translated
        by TrianMesh when this runs.
        """
        kind = region.get('type', 'bbox')
        v = self.mesh.v_p.to_numpy()

        if kind == 'bbox':
            lo = np.asarray(region['min'], dtype=np.float32)
            hi = np.asarray(region['max'], dtype=np.float32)
            return np.where(((v >= lo) & (v <= hi)).all(axis=1))[0]

        if kind == 'indices':
            # Direct vertex index list. Comes from YAML for testing, or from
            # metadata.json (RL hook, per-step index updates).
            src = region.get('values')
            if src is None and self.metadata:
                src = self.metadata.get('fixed_indices')
            if src is None:
                raise ValueError("fixed_region.type=indices requires `values` in YAML or `fixed_indices` in metadata.json")
            return np.asarray(src, dtype=np.int32)

        raise ValueError(f"Unknown fixed_region.type: {kind!r} (expected 'bbox' or 'indices')")

    def initialize_force_fields(self):
        """Initialize force field arrays"""
        self.pointForce = ti.Vector.field(3, dtype=ti.f32, shape=self.mesh.n_vert)
        self.pointForce_copy = ti.Vector.field(3, dtype=ti.f32, shape=self.mesh.n_vert)
        self.pointLocation_copy = ti.Vector.field(3, dtype=ti.f32, shape=self.mesh.n_vert)
        self.mesh_vp_copy = ti.Vector.field(3, dtype=ti.f32, shape=self.mesh.n_vert)
        self.mesh_vel_copy = ti.Vector.field(3, dtype=ti.f32, shape=self.mesh.n_vert)

        print("[IBMCloth] Initialized force fields")

    def setup(self):
        """Complete setup: mesh + constraints + forces"""
        self.initialize_mesh()
        self.initialize_constraints()
        self.initialize_fixed_points()
        self.initialize_force_fields()

        print("[IBMCloth] Setup complete")

    def get_solve_iters(self):
        """Get solver iterations from config"""
        return self.config['simulation'].get('solve_iters', 50)

    def get_dt(self):
        """Get time step from config"""
        return self.config['simulation'].get('dt', 0.0005)

    def get_gravity(self):
        """Get gravity vector from config"""
        gravity = self.config['simulation'].get('gravity', [0.0, 5.0, -3.0])
        return ti.Vector(gravity)


# Legacy compatibility: create global instance
# This allows existing ibm_cloth.py to work with minimal changes
def create_cloth_config(config_path: str = None, mesh_name: str = None):
    """Factory function to create IBM cloth configuration"""
    config = IBMClothConfig(config_path=config_path, mesh_name=mesh_name)
    config.setup()
    return config
