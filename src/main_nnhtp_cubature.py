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
from jax.example_libraries import optimizers
from scipy.optimize import nnls
from jax.numpy.linalg import norm
from tqdm import tqdm

import igl

# Imports from this project
import utils
import config
import fem_model
import system_utils

def main():
    # Build command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--system_name", type=str, required=True)
    parser.add_argument("--problem_name", type=str, required=True)
    
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--output_prefix', type=str, default='')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--pca_basis', type=str, required=True)
    parser.add_argument('--gradient_weighting', action='store_true')
    
    parser.add_argument('--n_X', type=int, required=True)
    parser.add_argument('--c_lazy', type=int, default=2000)
    parser.add_argument('--max_iter', type=int, default=20)
    
    parser.add_argument('--report_every', type=int, default=10)
    
    args = parser.parse_args()
    
    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())

    if args.gradient_weighting:
        print(f'WARNING: gradient weighting enabled')
    else:
        print(f'WARNING: gradient weighting disabled')

    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)
    Ne = system.mesh['E'].shape[0]
    
    def read_training_data(data_dir):
        dataset = []
        dmat_files = [filename for filename in os.listdir(data_dir) if filename.endswith('.dmat')]
        i = 0
        for file in dmat_files:
            filepath = os.path.join(data_dir, file)
            U = igl.read_dmat(filepath)
            n_vert = U.shape[0]
            dim = U.shape[1]
            U = U.reshape((n_vert * dim, ))
            dataset.append(U)
            
        return np.stack(dataset)
    
    print(f'Reading Training Data...')
    
    # [Nd, Nv*3]
    dataset = jnp.array(read_training_data(args.data_dir))

    # dataset_inds = jax.random.permutation(jax.random.PRNGKey(0), jnp.array(range(dataset.shape[0])))[:500]
    # dataset = dataset[dataset_inds, :]
    
    base_pos = system_def['init_pos']
    
    def remove_fixed_verts_and_move_to_origin(fixed_mask, initial_values):
        unfixed_values = initial_values[~fixed_mask]
        return unfixed_values + base_pos
    
    # [Nd, Nufixed*3]
    # [Nd, n]
    data_set = jax.vmap(partial(remove_fixed_verts_and_move_to_origin, system_def['fixed_mask']))(dataset)
    
    print(f'Reading pca basis from {args.pca_basis}')
    
    U = igl.read_dmat(os.path.join(args.pca_basis, "basis.dmat"))
    if args.gradient_weighting:
        Sigma = igl.read_dmat(os.path.join(args.pca_basis, "eigenvalues.dmat"))
    else:
        Sigma = np.ones(U.shape[1], dtype=float)
    
    U = jnp.array(U)
    Sigma = jnp.array(Sigma)
    
    # [n] -> [r]
    @jax.jit
    def pca_encoder(q):
        q = q - base_pos
        return q @ U
    
    # [r] -> [n]
    @jax.jit
    def pca_decoder(r):
        q = r @ U.T
        return q + base_pos
    
    # [n] -> [Nv, 3]
    def get_full_position(system_def, q):
            pos = system_utils.apply_fixed_entries( 
                    system_def['fixed_inds'], system_def['unfixed_inds'], 
                    system_def['fixed_values'], q).reshape(-1, 3)
            return pos
    
    # [Nv, 3] -> [1]
    @jax.jit
    def potential(system_def, z):
        q = pca_decoder(z)
        pos = get_full_position(system_def, q)
        return fem_model.fem_energy(system_def, system.mesh, fem_model.neohook_energy, pos)
    
    # pca latent samples
    # [Nd, r]
    z_list = jax.vmap(pca_encoder)(data_set)
    
    # [r] -> [r]
    def dP_dz(system_def, z):
        return jax.grad(partial(potential, system_def))(z)
    
    # latent gradients ground truth
    # [Nd, r]
    b = np.empty_like(z_list)
    print(f'{time.ctime()} == Building b...')
    for i in tqdm(range(z_list.shape[0])):
        b[i] = dP_dz(system_def, z_list[i]) * Sigma  # weight
    
    # [Nd]
    b_norm = jnp.linalg.norm(b, axis=1)
    b_norm_nonzero = jnp.where(b_norm < 1e-5, 1., b_norm)
    
    # normalized latent gradients ground truth
    b = np.array(jax.vmap(lambda x, y:x / y)(b, b_norm_nonzero))
    b = b.reshape(b.shape[0]*b.shape[1], )  # [Nd*r]
    b_norm = np.linalg.norm(b)
    
    # [r] -> [r]
    def dP_e_dz_normalized(e, z, g_norm):
        my_system_def = system_def.copy()
        my_system_def['cubature_inds'] = jnp.array([e])
        my_system_def['cubature_weights'] = jnp.array([1.])
        return Sigma * dP_dz(my_system_def, z) / g_norm
    
    @jax.jit
    def all_data_dP_e_dz_normalized(e):
        # [Nd, r]
        g = jax.vmap(partial(dP_e_dz_normalized, e))(z_list, b_norm_nonzero)
        return g.reshape(z_list.shape[0]*z_list.shape[1], )  # [Nd*r]
    
    _lazy_A = np.empty((z_list.shape[0]*z_list.shape[1], args.n_X + args.c_lazy), dtype=float)
    lazy_e_list = []
    def update_lazy_A(new_lazy_e_list):
        print(f'{time.ctime()} == updating lazy A...')
        nonlocal lazy_e_list
        lazy_e_list = new_lazy_e_list
        for e in tqdm(range(len(lazy_e_list))):
            _lazy_A[:, e] = all_data_dP_e_dz_normalized(jnp.asarray(lazy_e_list[e]))
    
    def shrink_lazy_A(lazy_e_sub_list):
        nonlocal lazy_e_list
        seleted_cols = []
        for e in lazy_e_sub_list:
            seleted_cols.append(np.searchsorted(lazy_e_list, e))
        _lazy_A[:, :len(lazy_e_sub_list)] = _lazy_A[:, seleted_cols]
        lazy_e_list = lazy_e_sub_list
    
    def get_lazy_A():
        return _lazy_A[:, :len(lazy_e_list)]
    
    def calulate_lazy_A_w(_w):
        return get_lazy_A() @ _w[lazy_e_list]
    
    def recover_full_vec_like_w(e_list, w_list):
        df_dw = np.zeros(Ne)
        df_dw[e_list] = w_list
        return df_dw
    
    def sparse_positive_projection(w_like_list, allowed_ind_list):
        w_like_list = np.where(w_like_list < 0, 0., w_like_list)  # positive projection
        w_like_list_on_allowed_inds = w_like_list[allowed_ind_list]
        selected_allowed_ind_list = np.argsort(w_like_list_on_allowed_inds)[::-1][:args.n_X]
        projected_e_list = allowed_ind_list[selected_allowed_ind_list]
        return projected_e_list, recover_full_vec_like_w(projected_e_list, w_like_list[projected_e_list])

    def lazy_df_dw(_w):
        partial_df_dw = 2 * (get_lazy_A().T @ (get_lazy_A() @ _w[lazy_e_list]) - get_lazy_A().T @ b)
        return recover_full_vec_like_w(lazy_e_list, partial_df_dw)
    
    def relative_error(_w):
        r = b - get_lazy_A() @ _w[lazy_e_list]
        return np.linalg.norm(r) / b_norm
    
    history = []
    def save_current_data(_w):    
        nonzero_mask = np.nonzero(_w)[0]
        _X = np.array(range(Ne))[nonzero_mask]
        _w_on_X = w[_X]
        print(f'Selected {len(_X)} non-zero cubatures.')
        data_dir = args.output_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        # save indices
        igl.write_dmat(os.path.join(data_dir, f"{args.output_prefix}indices.dmat"), np.array(_X))
        # save weights
        igl.write_dmat(os.path.join(data_dir, f"{args.output_prefix}weights.dmat"), np.array(_w_on_X))
        print(f'Write cubature indices and weights to {data_dir}')
        with open(os.path.join(data_dir, f'{args.output_prefix}history.json'), 'w') as f:
            json.dump(history, f, indent=4)

    # prepare init weights
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    X = jax.random.permutation(subkey, jnp.array(range(Ne)))[:args.n_X]
    update_lazy_A(X)
    print(f'{time.ctime()} == nnls...')
    w_on_X, _ = nnls(get_lazy_A(), b)
    print(f'{time.ctime()} == nnls finished.')
    w = recover_full_vec_like_w(X, w_on_X)
    rel_error = relative_error(w)
    print(f'rel error: {rel_error}')
    history.append({'iter': 0, 'rel error': f'{rel_error}', 'curr_time': f'{time.ctime()}'})
    save_current_data(w)

    def generate_lazy_candidates(_X):
        all_e_list = np.array(range(Ne))
        candidate_e_list = np.delete(all_e_list, _X)
        nonlocal key
        key, subkey = jax.random.split(key)
        _C = jax.random.permutation(subkey, candidate_e_list)[:args.c_lazy]
        return _C
    
    for i in range(args.max_iter):
        print(f'Begin NN-HTP iter {i+1} of {args.max_iter}...')

        print(f'{time.ctime()} == evaluating lazy gradients of current w...')
        C = generate_lazy_candidates(X)
        new_lazy_e_list = np.sort(np.append(X, C))
        update_lazy_A(new_lazy_e_list)
        df_dw = lazy_df_dw(w)
        print(f'{time.ctime()} == evaluating lazy gradients finished.')

        df_dw_C = np.zeros_like(df_dw)
        df_dw_C[C] = df_dw[C]
        selected_C, _ = sparse_positive_projection(-df_dw_C, C)
        S = np.sort(np.append(X, selected_C))

        df_dw_S = np.zeros_like(df_dw)
        df_dw_S[S] = df_dw[S]
        def sqL2(vec):
            return np.sum(vec * vec)
        mu = sqL2(df_dw_S) / sqL2(calulate_lazy_A_w(df_dw_S))

        new_X, w = sparse_positive_projection(w - mu * df_dw_S, S)

        shrink_lazy_A(new_X)
        print(f'{time.ctime()} == nnls...')
        new_w_on_new_X, _ = nnls(get_lazy_A(), b)
        print(f'{time.ctime()} == nnls finished.')

        if np.all(X == new_X):
            print('Cubature set fixed, converged.')
            break
        
        X = new_X
        w = recover_full_vec_like_w(new_X, new_w_on_new_X)
        rel_error = relative_error(w)
        print(f'rel error: {rel_error}')
        history.append({'iter': i+1, 'rel error': f'{rel_error}', 'curr_time': f'{time.ctime()}'})
        save_current_data(w)

if __name__ == '__main__':
    main()
