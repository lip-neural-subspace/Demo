import sys, os
import time
from functools import partial
import argparse, json

import numpy as np
import scipy
import scipy.optimize
import jax
import jax.numpy as jnp
import jax.scipy

import equinox as eqx

import polyscope as ps
import polyscope.imgui as psim

import minimize
import igl

# Imports from this project
import utils
import config, layers, integrators, subspace

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")


def main():

    # Build command line arguments
    parser = argparse.ArgumentParser()

    # Shared arguments
    config.add_system_args(parser)
    config.add_jax_args(parser)

    # Arguments specific to this program
    parser.add_argument("--integrator", type=str, default = "implicit-proximal")
    parser.add_argument("--subspace_model", type=str)
    
    parser.add_argument("--interaction_stiffness", type=float, default=80.)
    parser.add_argument("--interaction_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_every", type=int, default=1)
    
    # Parse arguments
    args = parser.parse_args()

    # Process args
    config.process_jax_args(args)

    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())

    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)
    
    # Initialize polyscope
    ps.init()
    ps.set_ground_plane_mode('none')

    #########################################################################
    ### Load subspace map (if given)
    #########################################################################

    # If we're running on a use_subspace system, load it
    subspace_model_params = None
    subspace_dim = -1
    subspace_domain_dict = None
    if args.subspace_model:
        print(f"Loading subspace from {args.subspace_model}")

        # load subspace weights
        with open(args.subspace_model + '.json', 'r') as json_file:
            subspace_model_spec = json.loads(json_file.read())
        subspace_model = layers.create_model(subspace_model_spec)
        _, subspace_model_static = eqx.partition(subspace_model, eqx.is_array)

        subspace_model_params = eqx.tree_deserialise_leaves(args.subspace_model + ".eqx",
                                                            subspace_model)

        # load other info
        d = np.load(args.subspace_model + "_info.npy", allow_pickle=True).item()

        subspace_dim = d['subspace_dim']
        subspace_domain_dict = subspace.get_subspace_domain_dict(d['subspace_domain_type'])
        latent_comb_dim = system_def['interesting_states'].shape[0]
        t_schedule_final = d['t_schedule_final']

        def apply_subspace(subspace_model_params, x, cond_params):
            subspace_model = eqx.combine(subspace_model_params, subspace_model_static)
            return subspace_model(jnp.concatenate((x, cond_params), axis=-1),
                                  t_schedule=t_schedule_final)

        if args.system_name != d['system']:
            raise ValueError("system name does not match loaded weights")
        if args.problem_name != d['problem_name']:
            raise ValueError("problem name does not match loaded weights")
    use_subspace = subspace_model_params is not None

    print("System dimension: " + str(system_def['init_pos'].shape[0]))
    if use_subspace:
        print("Subspace dimension: " + str(subspace_dim))


    #########################################################################
    ### Set up state & UI params
    #########################################################################

    ## Integrator setup
    int_opts = {}
    int_state = {}
    integrators.initialize_integrator(int_opts, int_state, args.integrator)

    ## State of the system

    # UI state
    save_data = True
    run_sim = False
    eval_energy_every = True
    update_viz_every = True
    
    data_dir = args.output_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    base_pos = system.get_full_position(system, system_def, system_def['init_pos'])
    
    # load random interactions
    with open(args.interaction_json + '.json', 'r') as json_file: 
        interaction_json = json.loads(json_file.read())
    
    # write json to data_dir
    with open(data_dir + '/interaction.json', 'w') as f: 
        json.dump(interaction_json, f, indent=4)
        
    # interaction initial setting
    frame = -1
    i_interaction = 0
    spring_scale = system.mesh['scale'].max() / 2.
    system_def['k'] = args.interaction_stiffness
    mask = np.zeros((base_pos.shape[0], ))
    system_def['interaction_mask'] = mask
    system_def['force_dir'] = jnp.array([0., 0., 0.])
    system_def['target'] = jnp.ones(base_pos.shape)
    system_def['grab_r'] = interaction_json['config']['grab_radius']
    
    interaction_list = interaction_json['interactions']
    interaction_dict = interaction_list[0]
    n_frames = interaction_json['config']['n_frames']
    run_fixed_steps = 0
    
    # Set up state parameters
    if use_subspace:
        base_latent = jnp.zeros(subspace_dim) + subspace_domain_dict['initial_val']
    else:
        base_latent = None

    def reset_state():
        if use_subspace:
            int_state['q_t'] = base_latent
        else:
            int_state['q_t'] = system_def['init_pos']
        int_state['q_tm1'] = int_state['q_t']
        int_state['qdot_t'] = jnp.zeros_like(int_state['q_t'])

        system_def['interaction_mask'].fill(0)
        

        system.visualize(system_def, state_to_system(system_def, int_state['q_t']))

    def state_to_system(system_def, state):
        if use_subspace:
            return apply_subspace(subspace_model_params, state, system_def['cond_param'])
        else:
            # in the non-latent state, it's the identity
            return state

    if use_subspace:
        baseState = state_to_system(system_def, base_latent)
    else:
        baseState = system_def['init_pos']

    if use_subspace:
        subspace_fn = state_to_system
    else:
        subspace_fn = None

    ps.set_automatically_compute_scene_extents(False)
    reset_state()  # also creates initial viz

    print(f"state_to_system dtype: {state_to_system(system_def, int_state['q_t']).dtype}")

    @jax.jit
    def eval_potential_energy(system_def, q):
        return system.potential_energy(system_def, state_to_system(system_def, q))


    print(f'Simulation Frames: {n_frames}')
    print(f'Saved Frames: {int(n_frames / args.save_every)}')
    print(f"Click \"run {n_frames} steps\" to save trajectories")
    
    #########################################################################
    ### Main loop, sim step, and UI
    #########################################################################
    def main_loop():

        nonlocal int_opts, int_state, frame, mask, i_interaction, interaction_dict, run_sim, save_data, base_latent, update_viz_every, eval_energy_every, run_fixed_steps

        # Define the GUI

        # some latent sliders
        if use_subspace:

            psim.TextUnformatted(f"Subspace domain type: {subspace_domain_dict['domain_name']}")

            if psim.TreeNode("explore current latent"):

                psim.TextUnformatted("This is the current state of the system.")

                any_changed = False
                tmp_state_q = int_state['q_t'].copy()
                low = subspace_domain_dict['viz_entry_bound_low']
                high = subspace_domain_dict['viz_entry_bound_high']
                for i in range(subspace_dim):
                    s = f"latent_{i}"
                    val = tmp_state_q[i]
                    changed, val = psim.SliderFloat(s, val, low, high)
                    if changed:
                        any_changed = True
                        tmp_state_q = tmp_state_q.at[i].set(val)

                if any_changed:
                    integrators.update_state(int_opts, int_state, tmp_state_q, with_velocity=True)
                    integrators.apply_domain_projection(int_state, subspace_domain_dict)
                    system.visualize(system_def, state_to_system(system_def, int_state['q_t']))

                psim.TreePop()

        # Helpers to build other parts of the UI
        integrators.build_ui(int_opts, int_state)
        system.build_system_ui(system_def)

        # update visualization every frame
        if run_sim or run_fixed_steps:
            system.visualize(system_def, state_to_system(system_def, int_state['q_t']))

        # print energy
        if eval_energy_every:
            E = eval_potential_energy(system_def, int_state['q_t'])
            E_str = f"Potential energy: {E}"
            psim.TextUnformatted(E_str)

        _, eval_energy_every = psim.Checkbox("eval every", eval_energy_every)
        psim.SameLine()
        _, update_viz_every = psim.Checkbox("viz every", update_viz_every)

        if psim.Button("reset"):
            reset_state()
            frame = -1
            i_interaction = 0
            interaction_dict = interaction_list[0]

        psim.SameLine()

        if psim.Button("stop velocity"):
            integrators.update_state(int_opts, int_state, int_state['q_t'], with_velocity=False)

        psim.SameLine()

        _, run_sim = psim.Checkbox("run simulation", run_sim)
        psim.SameLine()
        _, save_data = psim.Checkbox("save trajectory", save_data)
        
        if psim.Button('Pre-Compile'):
            _, solver_info = integrators.timestep(system,
                                            system_def,
                                            int_state,
                                            int_opts,
                                            subspace_fn=subspace_fn,
                                            subspace_domain_dict=subspace_domain_dict)
            minimize.print_solver_info(solver_info)
            print('Pre-compile finished')
        
        psim.SameLine()
        
        if psim.Button(f"run {n_frames} steps"):
            run_fixed_steps = n_frames
        
        psim.SameLine()
        
        if run_sim or psim.Button("single step") or run_fixed_steps > 0:
            
            frame = frame + 1
            if save_data & (frame % args.save_every == 0):
                file_name = "displacements_" + f"{int(frame / args.save_every)}.dmat"
                save_path = os.path.join(data_dir, file_name)
                q = int_state['q_t']
                full_dist = system.get_full_position(system, system_def, q) - base_pos
                igl.write_dmat(save_path, np.array(full_dist), False)
            
            if frame == interaction_dict['t_start']:
                print(f'interaction {i_interaction} start')
                mask.fill(0)
                verts_dict = interaction_dict['grab_v']
                for i, w in verts_dict.items():
                    mask[int(i)] = w
                    
                system_def['interaction_mask'] = mask
                system_def['force_dir'] = jnp.array(interaction_dict['force_dir'])
                cur_pos = system.get_full_position(system, system_def, int_state['q_t'])
                system_def['target'] = cur_pos + spring_scale * system_def['force_dir']
            
            elif frame == interaction_dict['t_end']:
                print(f'interaction {i_interaction} end')
                mask.fill(0)
                system_def['interaction_mask'] = mask
                i_interaction += 1
                if i_interaction >= len(interaction_list):
                    i_interaction -= 1
                interaction_dict = interaction_list[i_interaction]
            
            
            # all-important timestep happens here
            int_state, solver_info = integrators.timestep(system,
                                            system_def,
                                            int_state,
                                            int_opts,
                                            subspace_fn=subspace_fn,
                                            subspace_domain_dict=subspace_domain_dict)
            if run_fixed_steps > 0:
                run_fixed_steps -= 1
            minimize.print_solver_info(solver_info)
            
    ps.set_user_callback(main_loop)
    ps.show()


if __name__ == '__main__':
    main()
