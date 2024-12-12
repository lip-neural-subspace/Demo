import jax
import jax.numpy as jnp
from functools import partial

import numpy as np
from tqdm import tqdm

import os
import argparse, json
import igl

import config

def compute_surface_vertices_and_normals(V, T, fixed_inds):
    surfaces = igl.boundary_facets(T)
    N = igl.per_vertex_normals(V, surfaces)
    
    surface_vert_mask = np.zeros((V.shape[0], ), dtype=bool)
    for i in range(surfaces.shape[0]):
        surface_vert_mask[surfaces[i]] = True
    
    surface_vert_mask[fixed_inds] = False
    
    print(f'Mesh pre-computation finished')
    return V[surface_vert_mask], N[surface_vert_mask]
    

# only sampling surface vertices
def sample_grab_points_and_force_directions(idx, n_sample, surface_v):
    sample_points = [idx]
    sample_force_dirs = [] 
    min_distances = np.zeros((surface_v.shape[0], ))
    min_distances.fill(999999)
    
    # fps
    for _ in range(n_sample - 1):
        distances = np.linalg.norm(surface_v - surface_v[sample_points[-1]], axis=1)
        min_distances = np.where(distances < min_distances, distances, min_distances)
        sample_points.append(np.argmax(min_distances))

    for _ in sample_points:
        force_dir = sample_force_dir_in_sphere()
        sample_force_dirs.append(force_dir)
    
    print(f'Sampling finished')
    return surface_v[sample_points], sample_force_dirs

def normalize(n:np.ndarray):
    if np.linalg.norm(n) == 0:
        print('err')
    return n / np.linalg.norm(n)


def sample_force_dir_in_sphere():
    while True:
        local_coord = np.random.uniform(-1, 1, (3,))
        if np.linalg.norm(local_coord) <= 1:
            break
    return normalize(local_coord).tolist()

def get_grab_point_and_weight(p, r, V):
    def grab_weights(center, radius, vert):
        d = jnp.linalg.norm(vert - center, ord=2)
        return 1. - d / radius
    
    grab_ws = jax.vmap(partial(grab_weights, p, r))(V)
    
    v_dict = {}
    sum = 0
    for i in range(grab_ws.shape[0]):
        if grab_ws[i] < 0.0:
            continue
        v_dict[f'{i}'] = float(grab_ws[i])
        sum += float(grab_ws[i])
    
    for key in v_dict.keys():
        v_dict[key] /= sum
    
    return v_dict

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--system_name", type=str, required=True)
    parser.add_argument("--problem_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    parser.add_argument("--grab_radius", type=float, default=None)
    parser.add_argument("--n_sample", type=int, default=30)
    parser.add_argument("--sample_type", type=str, default='fps')
    
    parser.add_argument("--init_sample_points", type=str, default=None)

    parser.add_argument("--n_actuation_frames", type=int, default=20)
    parser.add_argument("--n_empty_frames", type=int, default=20)
    parser.add_argument("--n_end_frames", type=int, default=100)

    # Parse arguments
    args = parser.parse_args()
    
    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)
    
    grab_radius = system.mesh['scale'].max() / 15
    if args.grab_radius is not None:    
        grab_radius = args.grab_radius
    
    fixed_inds = system_def['fixed_inds'].reshape(-1, 3)[:, 0] / 3
    fixed_inds = fixed_inds.astype(jnp.int32)
    if args.init_sample_points is None:
        surface_v, _ = compute_surface_vertices_and_normals(system.mesh['Vrest'], np.array(system.mesh['E']), fixed_inds)
        if args.sample_type == 'fps':
            print('Sample points by fps')
            n = range(surface_v.shape[0])
            id0 = np.random.choice(n, size=1)
            sample_points, force_dirs = sample_grab_points_and_force_directions(int(id0), args.n_sample, surface_v)
        else:
            print('Sample points by uniform random sampling')
            n = surface_v.shape[0]
            sample_list = []
            force_dirs = []
                
            for i in range(args.n_sample):
                sample_list.append(np.random.randint(0, n))
                force_dirs.append(sample_force_dir_in_sphere())
            
            sample_points = surface_v[sample_list]
    else:
        # load random interactions
        with open(args.init_sample_points + '.json', 'r') as json_file:
            sample_points_json = json.loads(json_file.read())
        sample_points_inds = sample_points_json['inds']
        sample_points = system.mesh['Vrest'][sample_points_inds]
        force_dirs = sample_points_json['force_dir']
        for i in range(len(force_dirs)):
            force_normalized = normalize(np.array(force_dirs[i]))
            force_dirs[i] = force_normalized.tolist()

    cur_frame = 0
    interaction_json = {}
    interaction_list = []

    n_actuation_frames = args.n_actuation_frames
    n_empty_frames = args.n_empty_frames
    n_end_frames = args.n_end_frames  
        
    for i in tqdm(range(sample_points.shape[0])):
        i_interaction_dict = {}
        v_dict = get_grab_point_and_weight(sample_points[i], grab_radius, system.mesh['Vrest'])
        force_dir = force_dirs[i]
        
        i_interaction_dict['grab_v'] = v_dict
        i_interaction_dict['force_dir'] = force_dir
        i_interaction_dict['t_start'] = cur_frame
        cur_frame += n_actuation_frames
        i_interaction_dict['t_end'] = cur_frame
        cur_frame += n_empty_frames
        
        interaction_list.append(i_interaction_dict)
    
    
    interaction_config = {'n_frames':cur_frame + n_end_frames, 'grab_radius':grab_radius}
    interaction_json['config'] = interaction_config
    interaction_json['interactions'] = interaction_list
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    with open(args.output_dir + 'interactions.json', 'w') as f:
        json.dump(interaction_json, f, indent=4)
    
    print('Done')
    
if __name__ == '__main__':
    main()