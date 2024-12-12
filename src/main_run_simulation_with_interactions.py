import math
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
import pandas as pd

import igl

# Imports from this project
import utils
import config, layers, integrators, subspace, minimize

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")


def main():

    # Build command line arguments
    parser = argparse.ArgumentParser()

    # Shared arguments
    config.add_jax_args(parser)
    parser.add_argument("--system_name", type=str, required=True)
    parser.add_argument("--problem_name", type=str, required=True)
    parser.add_argument("--timestep_h", type=float, default=0.05)
    parser.add_argument("--save_folder", type=str, default='simulation_trajectory')
    parser.add_argument("--save_timestep_h", type=float, default=0.05)
    parser.add_argument("--vis_timestep_h", type=float, default=0.2)

    # Arguments specific to this program
    parser.add_argument("--integrator", type=str, default = "implicit-proximal")

    # AE (leave both empty to disable AE)
    parser.add_argument("--latent_encoder", type=str)  # leave empty to disable init latent estimate
    parser.add_argument("--latent_decoder", type=str)  # this can be generated from encoder path
    
    # intermediate AE
    parser.add_argument('--encoder_T_path', type=str)
    parser.add_argument('--decoder_T_path', type=str)
    # PCA
    parser.add_argument("--pca_basis", type=str)  # leave empty to disable PCA
    # cubature
    parser.add_argument("--cubature_dir", type=str)  # leave empty to disable cubature
    
    # interactions json
    parser.add_argument("--interaction_json", type=str, required=True)
    parser.add_argument("--interaction_stiffness", type=float, default=80.)
    
    # DAE
    parser.add_argument("--dae_encoder", type=str)
    parser.add_argument("--dae_decoder", type=str)
    
    # Parse arguments
    args = parser.parse_args()

    # Process args
    config.process_jax_args(args)

    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())

    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)

    base_pos = system.get_full_position(system, system_def, system_def['init_pos'])
    # load random interactions
    with open(args.interaction_json + '.json', 'r') as json_file: 
        interaction_json = json.loads(json_file.read())
    
    if 'time_span' not in interaction_json['config']:
        print('WARNING: old interaction list version, assuming its timestep: 0.05')
        interaction_json['config']['time_span'] = interaction_json['config']['n_frames'] * 0.05
        for interaction in interaction_json['interactions']:
            interaction['t_start'] = interaction['t_start'] * 0.05
            interaction['t_end'] = interaction['t_end'] * 0.05
    interaction_json['config']['n_frames'] = int(round(interaction_json['config']['time_span'] / args.timestep_h))
    for interaction in interaction_json['interactions']:
        interaction['t_start'] = int(round(interaction['t_start'] / args.timestep_h))
        interaction['t_end'] = int(round(interaction['t_end'] / args.timestep_h))
    if 'target' not in interaction_json['interactions'][0]:
        print('WARNING: interaction list with only direction provided, please make sure this is the first run of this interaction')
    
    interaction_list = interaction_json['interactions']
    i_interaction = 0
    interaction_dict = interaction_list[0]
    
    spring_scale = system.mesh['scale'].max() / 2.5
    system_def['k'] = args.interaction_stiffness
    mask = np.zeros((base_pos.shape[0], ))
    system_def['interaction_mask'] = mask
    system_def['force_dir'] = jnp.array([0., 0., 0.])
    system_def['target'] = jnp.ones(base_pos.shape)
    system_def['grab_r'] = interaction_json['config']['grab_radius']
    
    frame = 0
    n_frames = interaction_json['config']['n_frames']
    
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
    print(f"Loading AE from {args.latent_encoder} and {args.latent_decoder}")

    # cubature loading
    if args.cubature_dir:
        print(f"Loading cubature from {args.cubature_dir}")
        indices = igl.read_dmat(os.path.join(args.cubature_dir, "indices.dmat"))
        weights = igl.read_dmat(os.path.join(args.cubature_dir, "weights.dmat"))
        system_def['cubature_inds'] = jnp.array(indices).astype(jnp.int32)
        system_def['cubature_weights'] = jnp.array(weights)
    
    # AE loading
    decoder_path, encoder_path = args.latent_decoder, args.latent_encoder
    if decoder_path is None and encoder_path is not None:
        decoder_path = args.latent_encoder.replace('encoder', 'decoder')
        print(f'generated decoder path from encoder path: {decoder_path}')
    if decoder_path is not None:
        # AE enabled, load decoder
        print(f"Loading AE decoder from {decoder_path}")
        with open(decoder_path + '.json', 'r') as json_file:
            decoder_spec = json.loads(json_file.read())
        decoder = layers.create_model(decoder_spec)
        decoder = eqx.tree_deserialise_leaves(decoder_path + ".eqx", decoder)
        if encoder_path is not None:
            # enabled, load encoder
            print(f"Loading AE encoder from {decoder_path}")
            with open(encoder_path + '.json', 'r') as json_file:
                encoder_spec = json.loads(json_file.read())
            encoder = layers.create_model(encoder_spec)
            encoder = eqx.tree_deserialise_leaves(encoder_path + ".eqx", encoder)
    
    # intermediate AE loading
    if args.encoder_T_path is not None:
        print(f"Loading intermediate AE encoder from {args.encoder_T_path}")
        with open(args.encoder_T_path + '.json', 'r') as json_file:
            encoder_T_spec = json.loads(json_file.read())
        encoder_T = layers.create_model(encoder_T_spec)
        encoder_T = eqx.tree_deserialise_leaves(args.encoder_T_path + ".eqx", encoder_T)
    if args.decoder_T_path is not None:
        print(f"Loading intermediate AE decoder from {args.decoder_T_path}")
        with open(args.decoder_T_path + '.json', 'r') as json_file:
            decoder_T_spec = json.loads(json_file.read())
        decoder_T = layers.create_model(decoder_T_spec)
        decoder_T = eqx.tree_deserialise_leaves(args.decoder_T_path + ".eqx", decoder_T)
    
    # load pca basis
    if args.pca_basis is not None:
        print(f"Loading PCA from {args.pca_basis}")
        U = igl.read_dmat(args.pca_basis + ".dmat")
        pca_latent_dim = U.shape[1]
        
        def filter(x):
            return x - U @ (x.T @ U).T     

    # DAE loading
    dae_decoder_path, dae_encoder_path = args.dae_decoder, args.dae_encoder
    if dae_decoder_path is None and dae_encoder_path is not None:
        dae_decoder_path = args.dae_encoder.replace('encoder', 'decoder')
        print(f'generated decoder path from encoder path: {dae_decoder_path}')
    if dae_decoder_path is not None:
        # AE enabled, load decoder
        print(f"Loading DAE decoder from {dae_decoder_path}")
        with open(dae_decoder_path + '.json', 'r') as json_file:
            dae_decoder_spec = json.loads(json_file.read())
        dae_decoder = layers.create_model(dae_decoder_spec)
        dae_decoder = eqx.tree_deserialise_leaves(dae_decoder_path + ".eqx", dae_decoder)
        if dae_encoder_path is not None:
            # enabled, load encoder
            print(f"Loading DAE encoder from {dae_encoder_path}")
            with open(dae_encoder_path + '.json', 'r') as json_file:
                dae_encoder_spec = json.loads(json_file.read())
            dae_encoder = layers.create_model(dae_encoder_spec)
            dae_encoder = eqx.tree_deserialise_leaves(dae_encoder_path + ".eqx", dae_encoder)
    
    
    # networks input to output mappings
    @jax.jit
    def encoder_mapping(q, cond_params):
        q = jnp.concatenate((q - system_def['init_pos'], cond_params), axis=-1)
        if dae_encoder_path is not None:
            z_a = dae_encoder(filter(q))
        if args.pca_basis is not None:
            q = q @ U
        if encoder_path is not None:
            q = encoder(q)
        if dae_encoder_path is not None:
            q = jnp.concatenate((q, z_a), axis=-1)
        if args.encoder_T_path is not None:
            q = encoder_T(q)
        return q
    
    @jax.jit
    def decoder_mapping(z, cond_params):
        z = jnp.concatenate((z, cond_params), axis=-1)
        if args.decoder_T_path is not None:
            z = decoder_T(z)
        if dae_decoder_path is not None:
            z_a = z[pca_latent_dim:]
            q_a = filter(dae_decoder(z_a))
            z = z[:pca_latent_dim]
        if decoder_path is not None:
            z = decoder(z)
        if args.pca_basis is not None:
            z = z @ U.T
        if dae_decoder_path is not None:
            z = z + q_a
        return z + system_def['init_pos']
    
    print("System dimension: " + str(system_def['init_pos'].shape[0]))
    print(f'Condition dimension: {system.cond_dim}')
    print('Encoder spec:')
    if args.pca_basis is not None:
        print(f'{U.shape[0]}->PCA->{U.shape[1]}')
        latent_dim = U.shape[1]
    if encoder_path is not None:
        print(f"{encoder_spec['in_dim']}->encoder->{encoder_spec['out_dim']}")
        latent_dim = encoder_spec['out_dim']
    if args.encoder_T_path is not None:
        print(f"{encoder_T_spec['in_dim']}->intermediate encoder->{encoder_T_spec['out_dim']}")
        latent_dim = encoder_T_spec['out_dim']
    print('Decoder spec:')
    if args.decoder_T_path is not None:
        print(f"{decoder_T_spec['in_dim']}->intermediate decoder->{decoder_T_spec['out_dim']}")
    if decoder_path is not None:
        print(f"{decoder_spec['in_dim']}->decoder->{decoder_spec['out_dim']}")
    if args.pca_basis is not None:
        print(f'{U.shape[1]}->PCA->{U.shape[0]}')
    if args.cubature_dir:
        print(f"cubature count: {len(system_def['cubature_inds'])}")


    #########################################################################
    ### Set up state & UI params
    #########################################################################

    ## Integrator setup
    int_opts = {'timestep_h': args.timestep_h}
    int_state = {}
    integrators.initialize_integrator(int_opts, int_state, args.integrator)

    ## State of the system

    # UI state
    run_sim = False
    eval_energy_every = True
    update_viz_every = False
    run_fixed_steps = 0
    save_traj = False
    n_saved_frames = 0
    last_vis_frame = frame
    last_vis_t = time.time()

    # Set up state parameters
    base_latent = encoder_mapping(system_def['init_pos'], system_def['cond_param'])

    @jax.jit
    def eval_potential_energy(system_def, q):
        return system.potential_energy(system_def, state_to_system(system_def, q))

    def reset_state():
        int_state['q_t'] = base_latent
        int_state['q_tm1'] = int_state['q_t']
        int_state['qdot_t'] = jnp.zeros_like(int_state['q_t'])
        int_state['potential'] = eval_potential_energy(system_def, int_state['q_t'])
        
        system_def['interaction_mask'].fill(0)
        system.visualize(system_def, state_to_system(system_def, int_state['q_t']))

    def state_to_system(system_def, state):
        return decoder_mapping(state, system_def['cond_param'])

    baseState = state_to_system(system_def, base_latent)
    subspace_fn = state_to_system

    ps.set_automatically_compute_scene_extents(False)
    reset_state()  # also creates initial viz

    print(f"state_to_system dtype: {state_to_system(system_def, int_state['q_t']).dtype}")

    def save_run_history(history, path):
        keys = history[0].keys()
        data_dict = {}
        for k in keys:
                data_dict[k] = []
        for record in history:
            for k in keys:
                data_dict[k].append(record[k])
        df = pd.DataFrame(columns=keys)
        for k in keys:
            df[k] = data_dict[k]
        df.to_csv(path)
        
    #########################################################################
    ### Main loop, sim step, and UI
    #########################################################################
    simulation_history = []

    def main_loop():

        nonlocal int_opts, int_state, run_sim, save_traj, frame, n_saved_frames
        nonlocal mask, i_interaction, interaction_dict, base_latent
        nonlocal update_viz_every, eval_energy_every, simulation_history, run_fixed_steps
        nonlocal last_vis_t, last_vis_frame

        # Define the GUI
        if psim.TreeNode("explore current latent"):

            psim.TextUnformatted("This is the current state of the system.")

            any_changed = False
            tmp_state_q = int_state['q_t'].copy()
            low = -10
            high = 10
            for i in range(latent_dim):
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

        # print energy
        if eval_energy_every:
            E_str = f"Potential energy: {int_state['potential']}"
            psim.TextUnformatted(E_str)

        _, eval_energy_every = psim.Checkbox("eval every", eval_energy_every)
        psim.SameLine()
        _, update_viz_every = psim.Checkbox("viz every", update_viz_every)

        if psim.Button("reset"):
            reset_state()
            simulation_history = []
            frame = 0
            i_interaction = 0
            interaction_dict = interaction_list[0]
            last_vis_frame = frame
            last_vis_t = time.time()

        psim.SameLine()

        if psim.Button("stop velocity"):
            integrators.update_state(int_opts, int_state, int_state['q_t'], with_velocity=False)

        psim.SameLine()
        
        if psim.Button("Save History"):
            utils.ensure_dir_exists(args.save_folder)
            save_path = os.path.join(args.save_folder, f'simulation_history.csv')
            save_run_history(simulation_history, save_path)
            print(f'History saved to {save_path}.')
        
        psim.SameLine()
        _, save_traj = psim.Checkbox("save trajectory", save_traj)
        
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

        _, run_sim = psim.Checkbox("run simulation", run_sim)
        psim.SameLine()

        if psim.Button(f"single step"):
            run_fixed_steps = 1
        
        while run_sim or run_fixed_steps > 0:
            if save_traj and frame == 0:
                utils.ensure_dir_exists(args.save_folder)
                np.save(os.path.join(args.save_folder, 'elements.npy'), system.mesh['E'])
                np.save(
                    os.path.join(args.save_folder, f'pos_{n_saved_frames:04}.npy'), 
                    system.get_full_position(system, system_def, decoder_mapping(int_state['q_t'], system_def['cond_param'])))
                n_saved_frames += 1
            
            if frame == interaction_dict['t_start']:
                print(f'interaction {i_interaction} start')
                mask.fill(0)
                verts_dict = interaction_dict['grab_v']
                for i, w in verts_dict.items():
                    mask[int(i)] = w
                            
                system_def['interaction_mask'] = mask
                q = decoder_mapping(int_state['q_t'], system_def['cond_param'])
                cur_pos = system.get_full_position(system, system_def, q)
                representive_interaction_vert_id = np.argmax(system_def['interaction_mask'])
                representive_interaction_vert_pos = cur_pos[representive_interaction_vert_id]
                if 'target' not in interaction_dict:
                    force_dir = spring_scale * jnp.array(interaction_dict['force_dir'])
                    interaction_dict['target'] = (representive_interaction_vert_pos + force_dir).tolist()
                else:
                    force_dir = jnp.array(interaction_dict['target']) - representive_interaction_vert_pos
                system_def['target'] = cur_pos + force_dir
            
            elif frame == interaction_dict['t_end']:
                print(f'interaction {i_interaction} end')
                mask.fill(0)
                system_def['interaction_mask'] = mask
                i_interaction += 1
                if i_interaction >= len(interaction_list):
                    i_interaction -= 1
                    if save_traj:
                        with open(args.interaction_json + '.json', 'r') as fp: 
                            original_interaction_json = json.load(fp)
                        for i in range(len(interaction_list)):
                            original_interaction_json['interactions'][i]['target'] = interaction_list[i]['target']
                        with open(os.path.join(args.save_folder, 'interactions.json'), 'w') as fp: 
                            json.dump(original_interaction_json, fp)
                interaction_dict = interaction_list[i_interaction]
            
            int_state, solver_info = integrators.timestep(system,
                                            system_def,
                                            int_state,
                                            int_opts,
                                            subspace_fn=subspace_fn,
                                            subspace_domain_dict=subspace_domain_dict)
            if eval_energy_every:
                int_state['potential'] = eval_potential_energy(system_def, int_state['q_t'])
                solver_info['potential'] = int_state['potential']
            minimize.print_solver_info(solver_info)
            simulation_history.append(solver_info)
            
            save_to_sim_frame_rate = args.timestep_h / args.save_timestep_h
            closet_save_frame = round(frame * save_to_sim_frame_rate)
            if save_traj and \
               abs((frame-1) * save_to_sim_frame_rate - closet_save_frame) >= abs(frame * save_to_sim_frame_rate - closet_save_frame) and \
               abs((frame+1) * save_to_sim_frame_rate - closet_save_frame) >= abs(frame * save_to_sim_frame_rate - closet_save_frame):
                np.save(
                    os.path.join(args.save_folder, f'pos_{n_saved_frames:04}.npy'), 
                    system.get_full_position(system, system_def, decoder_mapping(int_state['q_t'], system_def['cond_param'])))
                n_saved_frames += 1
            
            frame += 1
            run_fixed_steps -= 1
            if math.floor(frame * args.timestep_h / args.vis_timestep_h) < math.floor((frame + 1) * args.timestep_h / args.vis_timestep_h):
                break

        # update visualization every frame
        if update_viz_every or run_sim or run_fixed_steps > 0:
            system.visualize(system_def, state_to_system(system_def, int_state['q_t']))
            psim.TextUnformatted(
                f'frame: {frame}, t: {frame * args.timestep_h:.2}, '\
                f'sim-to-real time rate: {(frame - last_vis_frame) * args.timestep_h / (time.time() - last_vis_t):.4}x')
            last_vis_frame = frame
            last_vis_t = time.time()

    ps.set_user_callback(main_loop)
    ps.show()


if __name__ == '__main__':
    main()
