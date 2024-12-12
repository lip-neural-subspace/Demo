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
    
    parser.add_argument('--max_n', type=int, required=True)
    parser.add_argument('--tol_error', type=float, required=True)
    parser.add_argument('--n_candidate', type=int, required=True)
    parser.add_argument('--init_selected_elements', type=str, required=False)
    
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
    
    # [N, Nv*3]
    dataset = jnp.array(read_training_data(args.data_dir))
    
    base_pos = system_def['init_pos']
    
    def remove_fixed_verts(fixed_mask, initial_values):
        unfixed_values = initial_values[~fixed_mask]
        return unfixed_values + base_pos
    
    # [N, Nufixed*3]
    data_set = jax.vmap(partial(remove_fixed_verts, system_def['fixed_mask']))(dataset)
    
    print(f'Reading pca basis from {args.pca_basis}')
    
    U = igl.read_dmat(os.path.join(args.pca_basis, "basis.dmat"))
    if args.gradient_weighting:
        Sigma = igl.read_dmat(os.path.join(args.pca_basis, "eigenvalues.dmat"))
    else:
        Sigma = np.ones(U.shape[1], dtype=float)
    
    U = jnp.array(U)
    Sigma = jnp.array(Sigma)
    # 3*Nufixed -> r
    @jax.jit
    def pca_encoder(q):
        q = q - base_pos
        return q @ U
    
    # r -> 3*Nufixed 
    @jax.jit
    def pca_decoder(r):
        q = r @ U.T
        return q + base_pos
    
    
    def get_full_position(system_def, q):
        pos = system_utils.apply_fixed_entries( 
                system_def['fixed_inds'], system_def['unfixed_inds'], 
                system_def['fixed_values'], q).reshape(-1, 3)
        return pos
    
    @jax.jit
    def potential(system_def, r):
        q = pca_decoder(r)
        pos = get_full_position(system_def, q)
        return fem_model.fem_energy(system_def, system.mesh, fem_model.neohook_energy, pos)
    
    # pca latent samples
    # [N, r]
    r_set = jax.vmap(pca_encoder)(data_set)
    
    def potential_grad(system_def, r):
        return jax.grad(partial(potential, system_def))(r)
    
    # A * w = b
    # b[N, r]
    b = np.empty((r_set.shape[0], r_set.shape[1]), dtype=float)
    
    for i in range(r_set.shape[0]):
        b[i] = potential_grad(system_def, r_set[i]) * Sigma  # weighted
    
    b_norm = jnp.linalg.norm(b, axis=1)
    b_norm_nonzero = jnp.where(b_norm < 1e-5, 1., b_norm)
    
    def grad_element_normalized(ind, r, g_norm):
        cond_system_def = system_def.copy()
        cond_system_def['cubature_inds'] = jnp.array([ind])
        cond_system_def['cubature_weights'] = jnp.array([1.])
        return Sigma * potential_grad(cond_system_def, r) / g_norm
    
    @jax.jit
    def grad_element(ind):
        # [N, r]
        g = jax.vmap(partial(grad_element_normalized, ind))(r_set, b_norm_nonzero)
        return g.reshape(r_set.shape[0]*r_set.shape[1], )
    
    # relative error
    b = jax.vmap(lambda x, y:x / y)(b, b_norm_nonzero)
    b = b.reshape(b.shape[0]*b.shape[1], )
    
    # initial setting
    key = jax.random.PRNGKey(0)
    n_total_elements = system.mesh['E'].shape[0]
    candidate_set_size = args.n_candidate
    ind_list = jax.random.permutation(key, jnp.array(range(n_total_elements)))
    selected_set = []
    history = []
    
    if args.init_selected_elements is not None:
        # weights = igl.read_dmat(os.path.join(args.init_selected_elements, "weights.dmat"))
        indices = igl.read_dmat(os.path.join(args.init_selected_elements, "indices.dmat"))
        selected_set = indices.astype(int).tolist()
    
    A = np.empty((r_set.shape[0]*r_set.shape[1], args.max_n), dtype=float)
    for i in range(len(selected_set)):
        A[:, i] = grad_element(selected_set[i])
    
    def select_element(r, i_start):
        i_end = i_start + candidate_set_size
            
        candidates = ind_list[i_start:i_end]
        candidates_subset = np.empty((len(candidates), r_set.shape[0]*r_set.shape[1]), dtype=float)
        
        for i in range(len(candidates)):
            candidates_subset[i] = grad_element(candidates[i]) 
        
        # (id, value)
        best_element = [0, - jnp.inf]
        # select best element
        for i in range(len(candidates)):
            g = candidates_subset[i]
            g_norm = norm(g)
            if g_norm < 1e-5:
                continue
            e = jnp.dot(g, r) / ( g_norm * norm(r))
            if best_element[1] < e:
                best_element[1] = e
                best_element[0] = i
        
        return candidates[best_element[0]], candidates_subset[best_element[0]]
    
    def greedy_cubature(A, b, tol, max_n, select_e):
        n = len(select_e)
        if n > 0:
            As = A[:, :n]
            w, _ = nnls(As, b)
            r = b - As @ w
        else:
            r = b
        b_norm = norm(b)
        r_error = norm(r)/b_norm
        print(f'selected elements:{n}\nrelative error:{r_error}')
        info_dict = {}
        info_dict[f'{n}'] = float(r_error)
        history.append(info_dict)
        
        while (norm(r) / b_norm > tol) & (n < max_n):
            Si, Ai = select_element(r, n * candidate_set_size)
            select_e.append(Si)
            A[:, n] = Ai
            As = A[:, :n+1]
            w, _ = nnls(As, b)
            r = b - As @ w
            n += 1
            if n % args.report_every == 0:
                r_error = norm(r)/b_norm
                info_dict = {}
                info_dict[f'{n}'] = float(r_error)
                history.append(info_dict)
                print(f'selected elements:{n}\nrelative error:{r_error}')
                
                selected_set = jnp.array(select_e)
                nozero_mask = jnp.nonzero(w)[0]
                selected_set = selected_set[nozero_mask]
                W = w[nozero_mask]
                
                data_dir = args.output_dir
                if not os.path.exists(data_dir):
                    os.mkdir(data_dir)
                # save indices
                igl.write_dmat(os.path.join(data_dir, f"indices_{n}.dmat"), np.array(selected_set))
                # save weights
                igl.write_dmat(os.path.join(data_dir, f"weights_{n}.dmat"), np.array(W))
                print(f'Write cubature indices and weights to {data_dir}')
                
                with open(data_dir + '/history.json', 'w') as f:
                    json.dump(history, f, indent=4)
                
                
        print(f'finally selected elements:{n}\nrelative error:{norm(r)/b_norm}')
        return select_e, w
            
    selected_set, W = greedy_cubature(A, b, args.tol_error, args.max_n, selected_set)
    
    selected_set = jnp.array(selected_set)
    nozero_mask = jnp.nonzero(W)[0]
    selected_set = selected_set[nozero_mask]
    W = W[nozero_mask]
    
    data_dir = args.output_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # save indices
    igl.write_dmat(os.path.join(data_dir, "indices.dmat"), np.array(selected_set))
    # save weights
    igl.write_dmat(os.path.join(data_dir, "weights.dmat"), np.array(W))
    print(f'Write cubature indices and weights to {data_dir}')
    
    with open(data_dir + '/history.json', 'w') as f:
        json.dump(history, f, indent=4)
if __name__ == '__main__':
    main()
