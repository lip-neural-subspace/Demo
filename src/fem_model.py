import ftplib

import jax
import jax.numpy as jnp
from functools import partial
from jax import grad, jit, vmap

import numpy as np
import potpourri3d as pp3d

import os
from io import StringIO

import polyscope as ps
import polyscope.imgui as psim

import system_utils
import utils

import igl

import math

pi = math.pi

def linear_energy(system_def, FT, mesh):

    dim = mesh["Vrest"].shape[1]
    poisson = system_def['poisson']
    Y = system_def['Y']
    A = mesh["A"]

    mu = 0.5 * Y / (1.0 + poisson)
    lamb = (Y * poisson) / ((1.0 + poisson)*(1.0 - 2.0*poisson)) # Plane strain condition (thick)

    E = 0.5*(FT + jnp.swapaxes(FT,1,2)) - jnp.eye(dim)[None,:,:]
    energies = mu*(E * E).sum(axis=(1, 2)) + 0.5*lamb*E.trace(axis1=1, axis2=2)**2

    return (A*energies).sum()
    
def StVK_energy(system_def, FT, mesh):

    dim = mesh["Vrest"].shape[1]
    poisson = system_def['poisson']
    Y = system_def['Y']
    A = mesh["A"]

    mu = 0.5 * Y / (1.0 + poisson)
    lamb = (Y * poisson) / ((1.0 + poisson)*(1.0 - 2.0*poisson)) # Plane strain condition (thick)
    # lamb = (Y * poisson) / (1.0 - poisson*poisson) # Plane stress condition (thin)

    E = 0.5*(FT @ jnp.swapaxes(FT,1,2) - jnp.eye(dim)[None,:,:])
    energies = mu*(E * E).sum(axis=(1, 2)) + 0.5*lamb*E.trace(axis1=1, axis2=2)**2

    return (A*energies).sum()


def neohook_energy(system_def, FT, mesh):

    poisson = system_def['poisson']
    Y = system_def['Y']
    A = mesh["A"]

    if 'cubature_inds' in system_def:
        A = system_utils.get_cubature_elements(A, system_def['cubature_inds'])
    
    mu = 0.5 * Y / (1.0 + poisson)
    lamb = (Y * poisson) / ((1.0 + poisson)*(1.0 - 2.0*poisson)) # Plane strain condition (thick)
    # lamb = (Y * poisson) / (1.0 - poisson*poisson) # Plane stress condition (thin)


    dim = mesh["Vrest"].shape[1]
    lambRep = lamb + mu
    alpha = 1.0 + (mu / lambRep)

    FTn = (FT * FT).sum(axis=(1, 2))
    dec = jnp.linalg.det(FT) - alpha

    energies0 = 0.5*( mu*dim + lambRep*(1.0-alpha)*(1.0-alpha) )
    energies = 0.5*( mu*FTn + lambRep*dec*dec )

    E = A*(energies - energies0)
    if 'cubature_inds' in system_def:
        E = system_def['cubature_weights'] * E
    return E


def neohook_thin_energy(system_def, FT, mesh):

    poisson = system_def['poisson']
    Y = system_def['Y']
    A = mesh["A"]

    if 'cubature_inds' in system_def:
        A = system_utils.get_cubature_elements(A, system_def['cubature_inds'])
    
    mu = 0.5 * Y / (1.0 + poisson)
    # lamb = (Y * poisson) / ((1.0 + poisson)*(1.0 - 2.0*poisson)) # Plane strain condition (thick)
    lamb = (Y * poisson) / (1.0 - poisson*poisson) # Plane stress condition (thin)


    dim = mesh["Vrest"].shape[1]
    lambRep = lamb + mu
    alpha = 1.0 + (mu / lambRep)

    FTn = (FT * FT).sum(axis=(1, 2))
    dec = jnp.linalg.det(FT) - alpha

    energies0 = 0.5*( mu*dim + lambRep*(1.0-alpha)*(1.0-alpha) )
    energies = 0.5*( mu*FTn + lambRep*dec*dec )

    E = A*(energies - energies0)
    if 'cubature_inds' in system_def:
        E = system_def['cubature_weights'] * E
    return E

# [Ne, ]
def fem_energy_element(system_def, mesh, material_energy, V):
    E = mesh["E"]
    DTI = mesh["DTI"]

    DT = V[E[:, 1:]] - V[E[:, 0]][:,None]
    FT = DTI @ DT
    return material_energy(system_def, FT, mesh)

# [1, ]
def fem_energy(system_def, mesh, material_energy, V):
    DTI = mesh["DTI"]
    if 'cubature_inds' in system_def:
        E = mesh['E']
        E = system_utils.get_cubature_elements(E, system_def['cubature_inds'])
        DTI = system_utils.get_cubature_elements(DTI, system_def['cubature_inds'])
    else:
        E = mesh["E"]

    DT = V[E[:, 1:]] - V[E[:, 0]][:,None]
    FT = DTI @ DT
    return jnp.sum(material_energy(system_def, FT, mesh))


def mean_strain_metric(system_def, mesh, V):

    dim = mesh["Vrest"].shape[1]

    E = mesh["E"]
    DTI = mesh["DTI"]
    A = mesh["A"]

    DT = V[E[:, 1:]] - V[E[:, 0]][:,None]
    FT = DTI @ DT

    E = 0.5*(FT @ jnp.swapaxes(FT,1,2) - jnp.eye(dim)[None,:,:])

    rigidity_density = jnp.sqrt((E * E).sum(axis=(1, 2)))

    return (A*rigidity_density).sum() / A.sum()

def tet_mesh_boundary_faces(tets):
    # numpy to numpy
    tets = np.array(tets)
    return igl.boundary_facets(tets)

def precompute_mesh(mesh):
   
    dim = mesh["Vrest"].shape[1]
    Vrest = mesh["Vrest"]
    E = mesh["E"]
    #Dm
    DT = Vrest[E[:, 1:]] - Vrest[E[:, 0]][:,None]
    #inv_Dm
    DTI = jnp.linalg.inv(DT)
    A = jnp.abs(jnp.linalg.det(DT)) / (dim * (dim-1))
    
    mesh["DTI"] = DTI
    mesh["A"] = A

    VA = jnp.zeros(Vrest.shape[0])
    Atile = jnp.tile(A, dim + 1).reshape((A.shape[0], dim + 1)) / (dim + 1)
    VA = VA.at[E].add(Atile)

    mesh["VA"] = VA

    if E.shape[-1] == 4:
        mesh['boundary_triangles'] = tet_mesh_boundary_faces(E)

    return mesh

def build_quad_mesh():

    mesh = {}

    mesh["Vrest"] = jnp.array([
                              [0., 0.],
                              [1., 0.],
                              [0., 1.],
                              [1., 1.]
                              ])
    mesh["E"] = jnp.array([
                          [0,1,2],
                          [1,3,2],
                          ])
    return mesh

def build_tet_mesh():

    mesh = {}

    p = np.array([1.0, 0., 0.])

    ang = np.pi*2.0/3.0
    R = np.array([[np.cos(ang),  0.0, np.sin(ang)], [0.0,  1.0, 0.0], [-np.sin(ang),  0.0, np.cos(ang)]])

    Rp = np.matmul(R, p)
    RRp = np.matmul(R, Rp)

    mesh["Vrest"] = np.array([
                              [0., 1.2, 0.],
                              p,
                              Rp,
                              RRp
                              ])
    mesh["E"] = np.array([
                          [0,1,2,3]
                          ])

    return mesh

def load_obj(filename):

    verts, faces = pp3d.read_mesh(filename)
    
    mesh = {}
    mesh["Vrest"] = verts
    mesh["E"] = jnp.array(faces)

    return mesh

def load_tri_mesh( file_name_root ):

    # ele file
    lines = open(file_name_root+".ele").readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    n_tri, n_nodesPerTriangle, n_attribT = map(int, lines[0].split())
    faces = np.loadtxt(StringIO(''.join(lines[1:])), dtype=int)[:,1:4]

    regions = 0
    if n_attribT > 0:
        regions = np.loadtxt(StringIO(''.join(lines[1:])), dtype=int)[:,4]

    # node file
    lines = open(file_name_root+".node").readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    n_vert, n_dim, n_attrib, n_bmark = map(int, lines[0].split())
    verts = np.loadtxt(StringIO(''.join(lines[1:])), dtype=float)[:,1:3]

    center = (verts.max(axis = 0) + verts.min(axis = 0))/2
    scale = verts.max(axis = 0) - verts.min(axis = 0)
    verts = (verts - center)/scale.max() 

    first_node_index = np.loadtxt(StringIO(''.join(lines[1])), dtype=int)[0]
    faces = faces - first_node_index

    #
    mesh = {}
    mesh["Vrest"] = verts
    mesh["E"] = jnp.array(faces)
    if n_attribT > 0:
        Y = np.full( (1, n_tri), 1.5e3 )
        np.put( Y, np.argwhere(regions[:] == 2), 1.5e2 )
        poisson = np.full( (1, n_tri), 0.4 )
        np.put( poisson, np.argwhere(regions[:] == 2), 0.4 )
        mesh["Y"] = Y
        mesh["poisson"] = poisson
        mesh["regions"] = regions

    return mesh

def load_tet_mesh_igl(file_name_root, normalize=True):
    verts, tets, _ = igl.read_mesh( file_name_root )
    scale = verts.max(axis = 0) - verts.min(axis = 0)
    
    mesh = {}
    mesh['scale'] = scale
    if normalize:
        center = (verts.max(axis = 0) + verts.min(axis = 0))/2
        verts = (verts - center)/scale.max()
        mesh['scale'] = scale / scale.max()

    mesh["Vrest"] = verts
    mesh["E"] = jnp.array(tets)
    
    return mesh


def load_tet_mesh( file_name_root, normalize=True ):

    # ele file
    lines = open(file_name_root+".ele").readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    n_tri, n_nodesPerTriangle, n_attribT = map(int, lines[0].split())
    data = np.loadtxt(StringIO(''.join(lines[1:])), dtype=int)[:,1:]
    tets = data[:,0:4]
    regions = 0
    if n_attribT > 0:
        regions = data[:,-1]

    # node file
    lines = open(file_name_root+".node").readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    n_vert, n_dim, n_attrib, n_bmark = map(int, lines[0].split())
    verts = np.loadtxt(StringIO(''.join(lines[1:])), dtype=float)[:,1:4]

    first_node_index = np.loadtxt(StringIO(''.join(lines[1])), dtype=int)[0]
    tets = tets - first_node_index
    
    mesh = {}
    scale = verts.max(axis = 0) - verts.min(axis = 0)
    mesh['scale'] = scale
    if normalize:
        center = (verts.max(axis = 0) + verts.min(axis = 0))/2
        verts = (verts - center)/scale.max()
        mesh['scale'] = scale / scale.max()
        
    mesh["Vrest"] = verts
    mesh["E"] = jnp.array(tets)
    if n_attribT > 0:
        Y = np.full((1, n_tri), 100e2)
        np.put(Y, np.argwhere(regions[:] == 0), 1e2)
        poisson = np.full((1, n_tri), 0.4)
        np.put(poisson, np.argwhere(regions[:] == 0), 0.4)
        mesh["Y"] = Y
        mesh["poisson"] = poisson
        mesh["regions"] = regions

    return mesh


###

class FEMSystem:

    def __init__(self):
        
        self.mesh = None
    
    @staticmethod
    def construct(problem_name):

        system_def = {}
        system = FEMSystem()

        system.system_name = "FEM"
        system.problem_name = str(problem_name)

        # set some defaults
        system_def['external_forces'] = {}
        system_def['boundary_conditions'] = {}
        system_def['cond_param'] = jnp.zeros((0,))
        system.cond_dim = 0
       
        def get_full_position(self, system_def, q):
            pos = system_utils.apply_fixed_entries( 
                    system_def['fixed_inds'], system_def['unfixed_inds'], 
                    system_def['fixed_values'], q).reshape(-1, self.pos_dim)
            return pos
        system.get_full_position = get_full_position
        
        def update_conditional(self, system_def):
            return system_def # default does nothing
        system.update_conditional = update_conditional
            
        system.material_energy = neohook_energy
    


        if problem_name.startswith('bistable'):

            mesh = load_tri_mesh( os.path.join(".", "data", "longerCantileverP2" ) )
            mesh["Vrest"][:,1] = 0.5 * mesh["Vrest"][:,1]

            mesh = precompute_mesh(mesh)
            system.material_energy = neohook_energy
            # if problem_name == 'bistable':
            system_def["gravity"] = jnp.array([0., 0.])
            # elif problem_name == 'bistable-gravity':
            # system_def["gravity"] = jnp.array([0., -0.98])
            system_def['poisson'] = jnp.array(0.45)
            system_def['Y'] = jnp.array(1e3)
            system_def['density'] = jnp.array(10.0)

            verts = mesh["Vrest"]

            verts_compress = verts
            verts_compress[:,0] *= 0.8
            verts_init = verts*0.8
            
            verts = jnp.array(verts)
            verts_compress = jnp.array(verts_compress)
            verts_init = jnp.array(verts_init)

            # identify verts that are on the x min and pin them.
            xmin = jnp.amin(verts, axis = 0)[0]
            xmax = jnp.amax(verts, axis = 0)[0]
            pinned_verts_mask = jnp.logical_or(verts[:,0] < (xmin + 1e-3),  verts[:,0] > xmax - 1e-3)
            pinned_verts_mask_flat = jnp.repeat(pinned_verts_mask,2)
            fixed_inds, unfixed_inds, fixed_values, unfixed_values = \
                system_utils.generate_fixed_entry_data(pinned_verts_mask_flat, verts_compress.flatten())
            system_def['fixed_mask'] = pinned_verts_mask_flat
            system_def["fixed_inds"] = fixed_inds
            system_def["unfixed_inds"] = unfixed_inds
            system_def["fixed_values"] = fixed_values

            # configure external forces
            xmid = (xmin + xmax) / 2
            system_def['external_forces']['force_verts_mask'] = jnp.logical_and((verts[:,0] > xmid - 1e-1), (verts[:,0] < xmid + 1e-1))
            system_def['external_forces']['pull_X'] = jnp.array(0.)
            system_def['external_forces']['pull_Y'] = jnp.array(0.)
            pull_minmax = (-0.1, 0.1)
            system_def['external_forces']['pull_strength_minmax'] = pull_minmax
            system_def['external_forces']['pull_strength'] = 0.5 * (pull_minmax[0] + pull_minmax[1])

            system.mesh = mesh
            system_def['init_pos'] = unfixed_values
            system.pos_dim = verts.shape[1]
            system.dim = system_def['init_pos'].size

            system_def['damping'] = 1.
        
        
        elif problem_name == 'compress_bar':
            mesh = load_tet_mesh_igl(os.path.join(".", "data", "bar_5634.mesh" ))
            mesh = precompute_mesh(mesh)
            system.material_energy = neohook_energy
            system_def["gravity"] = jnp.array([0., -0.98, 0.])
            # system_def["gravity"] = jnp.array([0., 0., 0.])
            system_def['poisson'] = jnp.array(0.45)
            system_def['Y'] = jnp.array(2e3)
            system_def['density'] = jnp.array(100.0)

            verts = mesh["Vrest"]
            
            verts = jnp.array(verts)
            # verts_compress = jnp.array(verts_compress)

            # identify verts that are on the x min and pin them.
            xmin = jnp.amin(verts, axis = 0)[0]
            xmax = jnp.amax(verts, axis = 0)[0]
            pinned_verts_mask = jnp.logical_or(verts[:, 0] < (xmin + 1e-3), verts[:, 0] > (xmax - 1e-3))
            pinned_verts_mask_flat = jnp.repeat(pinned_verts_mask, 3)
            fixed_inds, unfixed_inds, fixed_values, unfixed_values = \
                system_utils.generate_fixed_entry_data(pinned_verts_mask_flat, verts.flatten())
            system_def["fixed_inds"] = fixed_inds
            system_def["unfixed_inds"] = unfixed_inds
            system_def["fixed_values"] = fixed_values
            system_def["fixed_mask"] = pinned_verts_mask_flat
            
            cond_mask = []
            for i in range(0, fixed_values.shape[0], 3):
                if fixed_values[i] > xmax - 1e-3:
                    cond_mask.append(i)
        
            system_def['boundary_conditions']['conditions_mask'] = jnp.array(cond_mask, dtype=int)
            system_def['boundary_conditions']['cap_x'] = xmax
            system_def['boundary_conditions']['original_cap_x'] = jnp.array((xmax, ))
            cap_minmax = (-0.2 + xmax, xmax + 0.2)
            system_def['boundary_conditions']['cap_x_minmax'] = cap_minmax

            system_def['boundary_conditions']['rotations_mask'] = jnp.array(cond_mask, dtype=int) // 3
            system_def['boundary_conditions']['original_angle_x'] = jnp.array((0., ))
            system_def['boundary_conditions']['angle_x'] = 0.
            system_def['boundary_conditions']['original_fixed_pos'] = fixed_values.reshape(-1, 3)
            system_def['boundary_conditions']['angle_x_minmax'] = (-2. * pi, 2. * pi)
            
            system.mesh = mesh
            system_def['init_pos'] = unfixed_values
            system.pos_dim = verts.shape[1]
            system.dim = system_def['init_pos'].size

            system_def['damping'] = 1.
            
        elif problem_name == 'elephant':
            mesh = load_tet_mesh_igl( os.path.join(".", "data", "elephant.mesh" ))
            mesh = precompute_mesh(mesh)
            system.material_energy = neohook_energy
            
            system_def["gravity"] = jnp.array([0., -0.98, 0.])
            system_def['poisson'] = jnp.array(0.47)
            system_def['Y'] = jnp.array(1.5e4)
            system_def['density'] = jnp.array(500.0)
            
            system_def['damping'] = 1.0
            
            
            verts = jnp.array(mesh["Vrest"])

            # identify verts that are on the x min and pin them.
            system_def['pin_center1'] = jnp.array([0.0, 0.264, -0.26])
            system_def['pin_radius'] = 0.15
            
            def is_pin(center1, radius, vert):
                d1 = jnp.linalg.norm(vert - center1, ord=2)
                return d1 <= radius 
            pinned_verts_mask = jax.vmap(partial(is_pin, system_def['pin_center1'], system_def['pin_radius']))(verts)
            # zmin = jnp.amin( verts, axis = 0 )[2]
            # pinned_verts_mask = verts[:,2] < zmin + 1e-2
            pinned_verts_mask_flat = jnp.repeat(pinned_verts_mask,3)
            fixed_inds, unfixed_inds, fixed_values, unfixed_values = \
                system_utils.generate_fixed_entry_data(pinned_verts_mask_flat, verts.flatten())
                
            system_def["fixed_mask"] = pinned_verts_mask_flat
            system_def["fixed_inds"] = fixed_inds
            system_def["unfixed_inds"] = unfixed_inds
            system_def["fixed_values"] = fixed_values
    
            system.mesh = mesh
            system_def['init_pos'] = unfixed_values
            system.pos_dim = verts.shape[1]
            system.dim = system_def['init_pos'].size
        
        elif problem_name == 'bunny':
            mesh = load_tet_mesh_igl( os.path.join(".", "data", "bunny.mesh" ))
            mesh = precompute_mesh(mesh)
            system.material_energy = neohook_energy
            
            system_def["gravity"] = jnp.array([0, -0.98, 0])
            system_def['poisson'] = jnp.array(0.47)
            system_def['Y'] = jnp.array(5.e3)
            system_def['density'] = jnp.array(10.0)
            
            system_def['damping'] = 1.0
            
            verts = jnp.array(mesh["Vrest"])

            # identify verts that are on the x min and pin them.
            ymin = jnp.amin( verts, axis = 0 )[1]
            pinned_verts_mask = verts[:,1] < ymin + 1.e-2
            
            pinned_verts_mask_flat = jnp.repeat(pinned_verts_mask,3)
            fixed_inds, unfixed_inds, fixed_values, unfixed_values = \
                system_utils.generate_fixed_entry_data(pinned_verts_mask_flat, verts.flatten())
                
            system_def["fixed_mask"] = pinned_verts_mask_flat
            system_def["fixed_inds"] = fixed_inds
            system_def["unfixed_inds"] = unfixed_inds
            system_def["fixed_values"] = fixed_values
            
            system.mesh = mesh
            system_def['init_pos'] = unfixed_values
            system.pos_dim = verts.shape[1]
            system.dim = system_def['init_pos'].size
        
        elif problem_name.startswith('dinosaur'):
            mesh = load_tet_mesh_igl( os.path.join(".", "data", "dinosaur.mesh" ))
            # mesh['Vrest'] = 0.25 * mesh['Vrest']
            mesh = precompute_mesh(mesh)
            system.material_energy = neohook_energy
            
            if problem_name == 'dinosaur':
                system_def["gravity"] = jnp.array([0., -0.98, 0.])
            elif problem_name == 'dinosaur-quasi-static':
                system_def["gravity"] = jnp.array([0., 0., 0.])
            system_def['poisson'] = jnp.array(0.47)
            system_def['Y'] = jnp.array(5.e4)
            system_def['density'] = jnp.array(100.0)
            
            system_def['damping'] = 1.0
            
            verts = jnp.array(mesh["Vrest"])

            # identify verts that are on the x min and pin them.
            system_def['pin_center1'] = jnp.array([0.0, -0.5, 0.12])
            system_def['pin_radius'] = 0.24
            
            
            def is_pin(center1, radius, vert):
                d1 = jnp.linalg.norm(vert - center1, ord=2)
                return d1 <= radius
            pinned_verts_mask = jax.vmap(partial(is_pin, system_def['pin_center1'], system_def['pin_radius']))(verts)
            # ymin = jnp.amin( verts, axis = 0 )[1]
            # pinned_verts_mask = verts[:,0] < ymin + 2e-2
            pinned_verts_mask_flat = jnp.repeat(pinned_verts_mask,3)
            fixed_inds, unfixed_inds, fixed_values, unfixed_values = \
                system_utils.generate_fixed_entry_data(pinned_verts_mask_flat, verts.flatten())
                
            system_def["fixed_mask"] = pinned_verts_mask_flat
            system_def["fixed_inds"] = fixed_inds
            system_def["unfixed_inds"] = unfixed_inds
            system_def["fixed_values"] = fixed_values
            
            system.mesh = mesh
            system_def['init_pos'] = unfixed_values
            system.pos_dim = verts.shape[1]
            system.dim = system_def['init_pos'].size
        
        else:
            raise ValueError("unrecognized system problem_name")
        
        system_def['interesting_states'] = system_def['init_pos'][None,:]

        return system, system_def


    # ===========================================
    # === Energy functions 
    # ===========================================

    def mean_strain(self, system_def, q):

        pos = self.get_full_position(self, system_def, q)

        return mean_strain_metric(system_def, self.mesh, pos)

    def potential_energy(self, system_def,  q):
        system_def = self.update_conditional(self, system_def)

        pos = self.get_full_position(self, system_def, q)

        mass_lumped = self.mesh["VA"] * system_def['density']

        contact_energy = 0

        gravity = system_def["gravity"]
        gravity_energy = -jnp.sum(pos * gravity[None,] * mass_lumped[:,None])
        
        ext_force_energy = jnp.array(0.)
        interaction_energy = 0
        
        if 'interaction_mask' in system_def:
            # with stiffness
            def spring_energy(k, p1, p2):
                d = p1 - p2
                return 0.5 * k * jnp.sum(d * d)    
            
            interaction_energy += jnp.sum(system_def['interaction_mask'] * jax.vmap(partial(spring_energy, system_def['k']))(system_def['target'], pos))
            
        if 'pull_X' in system_def['external_forces']:
            mask = system_def['external_forces']["force_verts_mask"]
            if pos.shape[1] == 2:
                dir_force = jnp.array([[1., 0.]])
            else:
                dir_force = jnp.array([[1., 0., 0.]])
            masked_force = mask[:,None] * dir_force * system_def['external_forces']['pull_X'] * system_def['external_forces']['pull_strength'] 
            ext_force_energy += jnp.sum(masked_force*pos)

        if 'pull_Y' in system_def['external_forces']:
            mask = system_def['external_forces']["force_verts_mask"]
            if pos.shape[1] == 2:
                dir_force = jnp.array([[0., 1.]])
            else:
                dir_force = jnp.array([[0., 1., 0.]])
            masked_force = mask[:,None] * dir_force * system_def['external_forces']['pull_Y'] * system_def['external_forces']['pull_strength'] 
            ext_force_energy += jnp.sum(masked_force*pos)

        if 'pull_Z' in system_def['external_forces']:
            mask = system_def['external_forces']["force_verts_mask"]
            if pos.shape[1] == 2:
                dir_force = jnp.array([[0., 0.]])
            else:
                dir_force = jnp.array([[0., 0., 1.]])
            masked_force = mask[:,None] * dir_force * system_def['external_forces']['pull_Z'] * system_def['external_forces']['pull_strength'] 
            ext_force_energy += jnp.sum(masked_force*pos)
        
        return fem_energy(system_def, self.mesh, self.material_energy, pos) + gravity_energy + contact_energy + ext_force_energy + interaction_energy
    
    def potential_energy_with_cubature(self, system_def, q):
        q = jnp.concatenate((q, system_def["fixed_values"]))
        pos = q.reshape(-1, self.pos_dim)
        return fem_energy(system_def, self.mesh, self.material_energy, pos)
    
    def kinetic_energy(self, system_def, q, q_dot):
        system_def = self.update_conditional(self, system_def)

        pos_dot = system_utils.apply_fixed_entries(
                    system_def['fixed_inds'], system_def['unfixed_inds'], 
                    0., q_dot).reshape(-1, self.pos_dim)

        mass_lumped = self.mesh["VA"] * system_def['density']

        return 0.5 * jnp.sum(mass_lumped * jnp.sum(jnp.square(pos_dot), axis=-1))
    
    # ===========================================
    # === Conditional systems
    # ===========================================

    def sample_conditional_params(self, system_def, rngkey, rho=1.):
        cond = jnp.zeros((0,))
        if self.problem_name == 'cube':
            low, high = system_def['boundary_conditions']['cap_x_minmax']
            cond = jax.random.uniform(rngkey, (1, ), minval=low, maxval=high)
        return cond

    # ===========================================
    # === Visualization routines
    # ===========================================


    def build_system_ui(self, system_def):

        if psim.TreeNode("system UI"):

            psim.TextUnformatted("External forces:")
            
            if "pull_X" in system_def["external_forces"]:
                pulling = system_def['external_forces']['pull_X']
                _, pulling = psim.Checkbox("pull_X", pulling)
                system_def['external_forces']['pull_X'] = jnp.where(pulling, 1., 0.)

            if "pull_Y" in system_def["external_forces"]:
                pulling = system_def['external_forces']['pull_Y']
                _, pulling = psim.Checkbox("pull_Y", pulling)
                system_def['external_forces']['pull_Y'] = jnp.where(pulling, 1., 0.)

            if "pull_Z" in system_def["external_forces"]:
                pulling = system_def['external_forces']['pull_Z']
                _, pulling = psim.Checkbox("pull_Z", pulling)
                system_def['external_forces']['pull_Z'] = jnp.where(pulling, 1., 0.)
            
            if "pull_strength" in system_def["external_forces"]:
                low, high = system_def['external_forces']['pull_strength_minmax']
                _, system_def['external_forces']['pull_strength'] = psim.SliderFloat("pull_strength", system_def['external_forces']['pull_strength'], low, high)
            
            if "cap_x" in system_def["boundary_conditions"]:
                low, high = system_def['boundary_conditions']['cap_x_minmax']
                _, cap_x = psim.SliderFloat("cap_x", system_def['boundary_conditions']['cap_x'], low, high)
                system_def['boundary_conditions']['cap_x'] = cap_x
                system_def['fixed_values'] = system_def['fixed_values'].at[system_def['boundary_conditions']['conditions_mask']].set(cap_x)
            
            if "angle_x" in system_def["boundary_conditions"]:
                low, high = system_def['boundary_conditions']['angle_x_minmax']
                _, angle_x = psim.SliderFloat("rotation_angle_x", system_def['boundary_conditions']['angle_x'], low, high)
                system_def['boundary_conditions']['angle_x'] = angle_x
                sin = jnp.sin(angle_x)
                cos = jnp.cos(angle_x)
                r_mat = jnp.array([[1., 0., 0.],
                                    [0., cos, sin],
                                   [0., -sin, cos]])
                
                mask = system_def['boundary_conditions']['rotations_mask']
                fixed_pos = system_def['boundary_conditions']['original_fixed_pos'].at[mask, 0].set(cap_x)
                rotated_pos = jax.vmap(lambda x: r_mat @ x)(fixed_pos[mask])
                new_pos = fixed_pos.at[mask].set(rotated_pos)
                
                system_def['fixed_values'] = new_pos.flatten()
                
            psim.TreePop()

    def visualize(self, system_def, q, prefix="", transparency=1.0):
        system_def = self.update_conditional(self, system_def)

        name = self.problem_name + prefix

        pos = self.get_full_position(self, system_def, q)

        elem_list = self.mesh['E']

        if self.pos_dim == 2:
            ps_elems = ps.register_surface_mesh(name + " mesh", pos, np.array(elem_list))
            if 'regions' in self.mesh.keys():
                regions = self.mesh['regions']
                ps_elems.add_scalar_quantity("material colors r", regions, defined_on='faces', enabled=True)
        else:
            if 'regions' in self.mesh.keys():
                ps_elems = ps.register_volume_mesh(name + " mesh", pos, np.array(elem_list))
                regions = self.mesh['regions']
                ps_elems.add_scalar_quantity("material colors r", regions, defined_on='cells', enabled=True)
            else:
                ps_elems = ps.register_surface_mesh(name + " mesh", pos, np.array(self.mesh['boundary_triangles']))
                if 'interaction_mask' in system_def:
                    representive_interaction_vert_id = np.argmax(system_def['interaction_mask'])
                    if system_def['interaction_mask'][representive_interaction_vert_id] > 0:
                        idx = np.argmax(system_def['interaction_mask'])
                        interaction_center = pos[idx]
                        interaction_target = system_def['target'][idx]
                        sphere_elem = ps.register_point_cloud('interaction ball', np.array((interaction_center, interaction_target)), radius=system_def['grab_r'], transparency=0.2, color=(1., 1., 1.))
                        force_elem = ps.register_curve_network('interaction force', np.array((interaction_center, interaction_target)), np.array(((0,1),)), color=(1., 0., 0.))
                    else:
                        sphere_elem = ps.register_point_cloud('interaction ball', np.zeros((0,3)), radius=system_def['grab_r'], transparency=0.2, color=(1., 1., 1.))
                        force_elem = ps.register_curve_network('interaction force', np.zeros((0,3)), np.zeros((0,2)), color=(1., 0., 0.))
                #ps_elems = ps.register_surface_mesh(name + " mesh", pos, np.array(elem_list))
                
            if (self.problem_name == 'dinosaur') | (self.problem_name == 'dinosaur-quasi-static') | (self.problem_name == 'elephant') :
                center = system_def['pin_center1'].reshape(1, 3)
                ps.register_point_cloud('pin zone', center, radius=system_def['pin_radius'], transparency=0.2, color=(1., 1., 1.))
        if(transparency < 1.):
            ps_elems.set_transparency(transparency)


        if self.problem_name == "bistable":
            s = 0.4
            w = 0.07
            h = 0.1
            quad_block_verts = np.array([
                [-s-w,-h],
                [-s  ,-h],
                [-s  ,+h],
                [-s-w,+h],
                [+s  ,-h],
                [+s+w,-h],
                [+s+w,+h],
                [+s  ,+h],
                ])
            ps_endcaps = ps.register_surface_mesh("endcaps", quad_block_verts, np.array([[0,1,2,3], [4,5,6,7]]), color=(0.7,0.7,0.7), edge_width=4.)
        
        
        return (ps_elems)

    def export(self, system_def, x, prefix=""):

        system_def = self.update_conditional(self, system_def)

        pos = self.get_full_position(self, system_def, x)
        tri_list = self.mesh['boundary_triangles']

        filename = prefix+f"{self.problem_name}_mesh.obj"

        utils.write_obj(filename, pos, np.array(tri_list))

    def visualize_set_nice_view(self, system_def, q):

        if self.problem_name == 'spot':
            ps.look_at((-1.2, 0.8, -1.8), (0., 0., 0.))

        ps.look_at((2., 1., 2.), (0., 0., 0.))
