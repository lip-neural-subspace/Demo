import math
import sys, os
from functools import partial
import subprocess

import argparse, json, time

import numpy as np
import scipy
import scipy.optimize

import jax
import jax.numpy as jnp
import jax.scipy
import jax.scipy.optimize
import jax.nn
from jax.example_libraries import optimizers

import equinox as eqx

import polyscope as ps
import polyscope.imgui as psim

import igl

# Imports from this project
import utils
import config
import layers
import subspace
import fem_model
import system_utils


SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, '..')

def main():
    # Build command line arguments
    parser = argparse.ArgumentParser()

    # Shared arguments
    config.add_system_args(parser)
    config.add_training_args(parser)
    config.add_jax_args(parser)

    # dir settings
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--output_prefix', type=str, default='')
    parser.add_argument('--data_dir', type=str)
    
    # AE settings
    parser.add_argument('--model_type', type=str, default='MLP-Layers')
    parser.add_argument('--activation', type=str, default='ELU')
    parser.add_argument('--encoder_hidden_layers', type=int, nargs='+', default=[200, 200])
    parser.add_argument('--decoder_hidden_layers', type=int, nargs='+', default=[200, 200])
    parser.add_argument('--subspace_latent_dim', type=int)
    parser.add_argument('--evaluation_batch_size', type=int, default=64)
    
    # pca setting
    parser.add_argument('--pca_basis_dir', type=str, default=None)
    
    # start with existing model
    parser.add_argument('--existing_encoder', type=str, default=None)
    parser.add_argument('--existing_decoder', type=str, default=None)

    # extra training settings
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_decoder', action='store_true')
    parser.add_argument('--grad_clip', action='store_true')
    parser.add_argument('--grad_clip_value', type=float, default=0.8)
    parser.add_argument('--n_epoch', type=int, default=20000)
    parser.add_argument('--n_save', type=int, default=10)
    
    # loss weights / style / params
    parser.add_argument('--weight_landscape_reg', type=float, default=None)
    parser.add_argument('--landscape_type', type=str, default='A')

    # cubature
    parser.add_argument('--cubature_dir', type=str)
    
    
    # Parse arguments
    args = parser.parse_args()

    # Process args
    config.process_jax_args(args)

    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())

    # some random state
    rngkey = jax.random.PRNGKey(0)

    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)
    full_system_dim = system.dim
    base_pos = system_def['init_pos']
    
    # Build an informative output name
    network_filename_base = f'{args.output_prefix}neural_subspaceAE_{args.system_name}_{args.problem_name}_dim{args.subspace_latent_dim}'
    if args.weight_landscape_reg is not None:
        network_filename_base += f'_landscape{args.landscape_type}{args.weight_landscape_reg}'
    
    utils.ensure_dir_exists(args.output_dir)
    
    #read training data
    data_dir = args.data_dir
    def read_training_data(data_dir):
        dataset = []
        dmat_files = [filename for filename in os.listdir(data_dir) if filename.endswith('.dmat')]
        for file in dmat_files:
            filepath = os.path.join(data_dir, file)
            U = igl.read_dmat(filepath)
            n_vert = U.shape[0]
            dim = U.shape[1]
            U = U.reshape((n_vert * dim, ))
            dataset.append(U)
            
        return np.stack(dataset)
    
    print(f'Reading Training Data...')
    dataset_full = read_training_data(data_dir)
    
    #deal with fixed verts
    def remove_fixed_verts(fixed_mask, initial_values):
        unfixed_values = initial_values[~fixed_mask]
        return unfixed_values + system_def['init_pos']
    
    dataset = jax.vmap(partial(remove_fixed_verts, system_def['fixed_mask']))(dataset_full)
    dataset = np.array(dataset)
    dataset_inds = range(dataset.shape[0])
    dataset_inds = list(dataset_inds)
    
    
    ae_outer_dim = full_system_dim
    # read pca basis
    if args.pca_basis_dir is not None:
        print(f'Reading pca basis from {args.pca_basis_dir}')
        U = igl.read_dmat(os.path.join(args.pca_basis_dir, "basis.dmat"))
        ae_outer_dim = U.shape[1]

    
    # load cubature datas(if given)
    if args.cubature_dir:
        indices = igl.read_dmat(os.path.join(args.cubature_dir, "indices.dmat"))
        weights = igl.read_dmat(os.path.join(args.cubature_dir, "weights.dmat"))
        system_def['cubature_inds'] = jnp.array(indices).astype(jnp.int32)
        system_def['cubature_weights'] = jnp.array(weights)
        
    
    # construct AE encoder
    if args.existing_encoder is None:
        print(f'Constructing encoder')
        rngkey, subkey = jax.random.split(rngkey)
        encoder_spec = {
            'model_type': args.model_type,
            'activation': args.activation,
            'in_dim': ae_outer_dim,
            'hidden_layer_sizes': args.encoder_hidden_layers,
            'out_dim': args.subspace_latent_dim
        }
        encoder = layers.create_model(encoder_spec, subkey)
    else:
        print(f'Loading encoder from {args.existing_encoder}')
        with open(args.existing_encoder + '.json', 'r') as json_file:
            encoder_spec = json.loads(json_file.read())
        encoder = layers.create_model(encoder_spec)
        encoder = eqx.tree_deserialise_leaves(args.existing_encoder + '.eqx', encoder)
    encoder_params, encoder_static = eqx.partition(encoder, eqx.is_array)

    # construct AE decoder
    if args.existing_decoder is None:
        print(f'Constructing decoder')
        latent_dim = args.subspace_latent_dim
        rngkey, subkey = jax.random.split(rngkey)
        decoder_spec = {
            'model_type': args.model_type,
            'activation': args.activation,
            'in_dim': latent_dim,
            'hidden_layer_sizes': args.decoder_hidden_layers,
            'out_dim': ae_outer_dim
        }
        decoder = layers.create_model(decoder_spec, subkey)
    else:
        print(f'Loading decoder from {args.existing_decoder}')
        with open(args.existing_decoder + '.json', 'r') as json_file:
            decoder_spec = json.loads(json_file.read())
        decoder = layers.create_model(decoder_spec)
        decoder = eqx.tree_deserialise_leaves(args.existing_decoder + '.eqx', decoder)
        decoder.linear_layers
    decoder_params, decoder_static = eqx.partition(decoder, eqx.is_array)
    
    @jax.jit
    def encoder_mapping(params, q):
        q = q - base_pos
        combined_encoder = eqx.combine(params[0], encoder_static)
        if args.pca_basis_dir is not None:
            q = q @ U
        return combined_encoder(q)
    
    @jax.jit
    def decoder_mapping(params, z):
        combined_decoder = eqx.combine(params[1], decoder_static)
        q = combined_decoder(z)
        if args.pca_basis_dir is not None:
            q = q @ U.T
        return q + base_pos

    # potential evaluation
    def eval_Epot_of_full_state(system_def, q):
        system_def = system_def.copy()
        E_pot = system.potential_energy(system_def, q)
        return E_pot
    
    def eval_Epot_of_latent_state(system_def, params, z):
        q = decoder_mapping(params, z)
        return eval_Epot_of_full_state(system_def, q)
    
    
    def eval_Epot_and_grad_and_hessian_wrt_latent_state(system_def, params, z):
        eval_Epot_of_z = lambda zz: eval_Epot_of_latent_state(system_def, params, zz)
        E_pot, gradient = jax.value_and_grad(eval_Epot_of_z)(z)
        hessian = jax.hessian(eval_Epot_of_z)(z)
        
        return E_pot, gradient, hessian

    # create optimizer
    print(f'Creating optimizer...')

    def step_func(i_iter):
        out = args.lr \
              * jnp.clip((i_iter + 1) / float(args.lr_warm_up_iters + 1), 0, 1) \
              * (args.lr_decay_frac ** (i_iter // args.lr_decay_every))
        return out

    opt = optimizers.adam(step_func)
    all_params = [encoder_params, decoder_params]
    if args.freeze_encoder:
        all_params[0] = None
    if args.freeze_decoder:
        all_params[1] = None
    
    all_params = tuple(all_params)
    opt_state = opt.init_fn(all_params)

    def get_model_params_from_opt_params(params):
        params = list(params)
        if args.freeze_encoder:
            params[0] = encoder_params
        if args.freeze_decoder:
            params[1] = decoder_params

        return params

    # loss define
    def batch_loss_fn(params, q_samples):
        #encoder_params, decoder_params, pca_encoder_params, pca_decoder_params
        full_params = get_model_params_from_opt_params(params)
        sampler_E_pots = jax.vmap(partial(eval_Epot_of_full_state, system_def))(q_samples)
        latent_samples = jax.vmap(partial(encoder_mapping, full_params))(q_samples)
        
        recon_q_samples = jax.vmap(partial(decoder_mapping, full_params))(latent_samples)    
        if args.weight_landscape_reg == 0:
            latent_samples = latent_samples[:args.evaluation_batch_size]
        
        E_pots, gradients, hessians = jax.vmap(partial(eval_Epot_and_grad_and_hessian_wrt_latent_state, system_def, full_params))(latent_samples)
        
        
        def norm2_2(vector):
            return jnp.sum(vector * vector)
        
        # reconstruction loss
        reconstruct_loss = jnp.mean(jax.vmap(norm2_2)(q_samples - recon_q_samples))
        
        # landscape loss
        if args.weight_landscape_reg is not None:
            if args.landscape_type == 'A':
                def hessians_dist_to_one(h):
                    # hessians: [B,d,d], h: [d,d], return: [B] 
                    h_delta = hessians - h[None, :, :]
                    return jnp.sum(h_delta * h_delta, axis=(-2, -1))
                hessians_d_mat = jax.vmap(hessians_dist_to_one)(hessians)
                landscape_loss = jnp.mean(hessians_d_mat)
            elif args.landscape_type == 'B':
                landscape_loss = jnp.mean(jax.vmap(norm2_2)(hessians))
            elif args.landscape_type == 'C':
                def hessians_sqdist_to_one(h):
                    # hessians: [B,d,d], h: [d,d], return: [B] 
                    h_delta = hessians - h[None, :, :]
                    return jnp.sum(h_delta * h_delta, axis=(-2, -1))
                def latents_sqdist_to_one(z):
                    # latents: [B,d], z: [d], return: [B] 
                    z_delta = latent_samples - z[None, :]
                    return jnp.clip(jnp.sum(z_delta * z_delta, axis=(-1)), a_min=1e-5)
            
                n_B = hessians.shape[0]
                mask = np.ones((n_B, n_B), dtype=bool)
                np.fill_diagonal(mask, False)
                hessians_sqdist_mat = jax.vmap(hessians_sqdist_to_one)(hessians)
                latents_sqdist_mat = jax.vmap(latents_sqdist_to_one)(latent_samples)
                sqslope_mat = hessians_sqdist_mat / latents_sqdist_mat
                landscape_loss = jnp.mean(sqslope_mat[mask])
            
            elif args.landscape_type == 'D':
                def gradients_sqdist_to_one(g):
                    # gradients: [B,d], g: [d], return: [B] 
                    g_delta = gradients - g[None, :]
                    return jnp.sum(g_delta * g_delta, axis=(-1))
                def latents_sqdist_to_one(z):
                    # latents: [B,d], z: [d], return: [B] 
                    z_delta = latent_samples - z[None, :]
                    return jnp.clip(jnp.sum(z_delta * z_delta, axis=(-1)), a_min=1e-5)
            
                n_B = gradients.shape[0]
                mask = np.ones((n_B, n_B), dtype=bool)
                np.fill_diagonal(mask, False)
                gradients_sqdist_mat = jax.vmap(gradients_sqdist_to_one)(gradients)
                latents_sqdist_mat = jax.vmap(latents_sqdist_to_one)(latent_samples)
                sqslope_mat = gradients_sqdist_mat / latents_sqdist_mat
                landscape_loss = jnp.mean(sqslope_mat[mask])
                
            elif args.landscape_type == 'E':
                def E_pots_sqdist_to_one(e):
                    # E_pots: [B,], e: float, return: [B] 
                    e_delta = E_pots - e
                    return e_delta * e_delta
                def latents_sqdist_to_one(z):
                    # latents: [B,d], z: [d], return: [B] 
                    z_delta = latent_samples - z[None, :]
                    return jnp.clip(jnp.sum(z_delta * z_delta, axis=(-1)), a_min=1e-5)
            
                n_B = E_pots.shape[0]
                mask = np.ones((n_B, n_B), dtype=bool)
                np.fill_diagonal(mask, False)
                E_pots_sqdist_mat = jax.vmap(E_pots_sqdist_to_one)(E_pots)
                latents_sqdist_mat = jax.vmap(latents_sqdist_to_one)(latent_samples)
                sqslope_mat = E_pots_sqdist_mat / latents_sqdist_mat
                landscape_loss = jnp.mean(sqslope_mat[mask])
            else:
                raise ValueError("invalid landscape loss type.")

        loss_dict = {}
        weight_dict = {}
        energy_dict = {}
        loss_dict['L_reconstruction'] = reconstruct_loss
        weight_dict['L_reconstruction'] = 1.
        if args.weight_landscape_reg is not None:
            if args.weight_landscape_reg > 0:
                loss_dict['L_landscape'] = landscape_loss
                weight_dict['L_landscape'] = args.weight_landscape_reg
            else:
                energy_dict['L_landscape'] = landscape_loss
            if args.landscape_type == 'C':
                latents_dist_mat = jnp.sqrt(latents_sqdist_mat)
                slope_mat = jnp.sqrt(sqslope_mat)
                energy_dict['latent_dis_min'] = jnp.min(latents_dist_mat[mask])
                energy_dict['latent_dis_max'] = jnp.max(latents_dist_mat[mask])
                energy_dict['latent_dis_avg'] = jnp.mean(latents_dist_mat[mask])
                energy_dict['latent_dis_std'] = jnp.std(latents_dist_mat[mask])
                energy_dict['Lipschitz2nd'] = jnp.max(slope_mat[mask])
        
          
        
        energy_dict['samples_E_pot'] = jnp.mean(sampler_E_pots)
        energy_dict['reconstructed_E_pot'] = jnp.mean(E_pots)

        # sum up a total loss (mean over batch)
        total_loss = 0.
        for k, v in loss_dict.items():
            total_loss += weight_dict[k] * v

        return total_loss, (loss_dict, weight_dict, energy_dict)
    
    @jax.jit
    def batch_loss_fn_and_grads(params, q_samples):
        (loss, (loss_dict, weight_dict, energy_dict)), grads = jax.value_and_grad(batch_loss_fn, has_aux=True)(params, q_samples)
        return (loss, (loss_dict, weight_dict, energy_dict)), grads
    
    @jax.jit
    def has_nan_in_tree(tree):
        leaves, _ = jax.tree_util.tree_flatten(tree)
        return jnp.any(jnp.array([jnp.any(jnp.isnan(x)) for x in leaves]))
    
    optimizer_step = jax.jit(opt.update_fn)
    
    print(f'Training...')

    # Parameters tracked for each stat round
    losses = []
    i_save = 0
    stats_term_sums = {}
    stats_term_sums_history = []
    time_history = []
    n_frame = dataset.shape[0]
    n_batch = math.ceil(n_frame / args.batch_size)
    save_stride = args.n_epoch / args.n_save
    ## Main training loop
    for i_epoch in range(args.n_epoch):
        np.random.shuffle(dataset_inds)
        for i in range(n_batch):
            #rngkey, subkey = jax.random.split(rngkey)
            opt_params = opt.params_fn(opt_state)
            start = args.batch_size * i
            end = start + args.batch_size
            end = min(end, n_frame)
            q_samples = dataset[dataset_inds[start:end], :]
            
            (loss, (loss_dict, weight_dict, energy_dict)), grads = batch_loss_fn_and_grads(opt_params, q_samples)
            if has_nan_in_tree(grads):
                print('nan in grads detected, skip this batch')
                continue

            opt_state = optimizer_step(i_epoch * n_batch + i, grads, opt_state)

            # track statistics
            loss = float(loss)
            losses.append(loss)

            for k in loss_dict:
                if k not in stats_term_sums:
                    stats_term_sums[k] = []
                stats_term_sums[k].append(float(loss_dict[k]))

            for k in energy_dict:
                if k not in stats_term_sums:
                    stats_term_sums[k] = []
                stats_term_sums[k].append(float(energy_dict[k]))

            # n_sum_total += batch_size

            if jnp.isnan(loss):
                #log_statistics()
                exit()
                
            def save_model(this_name, model_params, model_static, model_spec):

                network_filename_pre = os.path.join(args.output_dir, network_filename_base) + this_name

                print(f'Saving result to {network_filename_pre}')

                model = eqx.combine(model_params, model_static)
                eqx.tree_serialise_leaves(network_filename_pre + '.eqx', model)
                with open(network_filename_pre + '.json', 'w') as json_file:
                    json_file.write(json.dumps(model_spec))
                np.save(
                    network_filename_pre + '_info', {
                        'system': args.system_name,
                        'problem_name': args.problem_name,
                        'latent_dim': args.subspace_latent_dim,
                        'weight_landscape_reg': args.weight_landscape_reg
                    })

                print(f'  ...done saving')
        
        def log_statistics():
            print(f'\n== epoch {i_epoch} / {args.n_epoch}  ({100. * i_epoch / args.n_epoch:.2f}%) == {time.ctime()}')
            mean_loss = np.mean(np.array(losses))
            print(f'  Loss: {mean_loss:.6f}')
            for k in loss_dict:
                print(f'   {k:>30}: {weight_dict[k]} * {np.mean(np.array(stats_term_sums[k])):.6f}')
            print('  Stats:')
            for k in energy_dict:
                print(f'   {k:>30}: {np.mean(np.array(stats_term_sums[k])):.6f}')
            record = {'i_epoch': i_epoch}
            for k in stats_term_sums:
                record[k] = float(np.mean(np.array(stats_term_sums[k])))
            stats_term_sums_history.append(record)
            np.save(os.path.join(args.output_dir, network_filename_base) + '_history', stats_term_sums_history)
            time_history.append(time.ctime())
            np.save(os.path.join(args.output_dir, network_filename_base) + '_time_history', time_history)
            
        
        if i_epoch % args.report_every == 0:
            log_statistics()
            losses = []
            stats_term_sums = {}
        
        if int(math.floor(i_epoch / save_stride)) > int(math.floor((i_epoch - 1) / save_stride)):
            # save
            opt_params = opt.params_fn(opt_state)
            full_params = get_model_params_from_opt_params(opt_params)
            if not args.freeze_encoder:
                save_model(f'_encoder_save{i_save:04d}', full_params[0], encoder_static, encoder_spec)
            if not args.freeze_decoder:
                save_model(f'_decoder_save{i_save:04d}', full_params[1], decoder_static, decoder_spec)
            i_save += 1
            

    # save results one last time
    #log_statistics()
    opt_params = opt.params_fn(opt_state)
    full_params = get_model_params_from_opt_params(opt_params)
    save_model(f'_encoder_final', full_params[0], encoder_static, encoder_spec)
    save_model(f'_decoder_final', full_params[1], decoder_static, decoder_spec)


if __name__ == '__main__':
    main()
