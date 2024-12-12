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

import igl

# Imports from this project
import config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, '..')

def main():
    # Build command line arguments
    parser = argparse.ArgumentParser()

    # Shared arguments
    config.add_jax_args(parser)
    config.add_system_args(parser)
    
    # dir settings
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--output_prefix', type=str, default='')
    parser.add_argument('--data_dir', type=str)
    
    # pca settings
    parser.add_argument('--max_tol_error', type=float, default=0.001)
    parser.add_argument('--max_n_basis', type=int, default=300)
    
    # Parse arguments
    args = parser.parse_args()

    # Process args
    config.process_jax_args(args)

    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())

    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)

    #reading training data and pca
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
    
    print(f'Read Training Data...')
    dataset = read_training_data(data_dir)
    
    #deal with fixed verts
    def remove_fixed_verts(fixed_mask, initial_values):
        unfixed_values = initial_values[~fixed_mask]
        return unfixed_values
    
    datas = jax.vmap(partial(remove_fixed_verts, system_def['fixed_mask']))(dataset)
    datas = np.array(datas)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    def PCA(samples, max_tol_error, max_n_basis, components=None, mem_=None):
        print('Doing SVD...')
        _, S, components =np.linalg.svd(samples, full_matrices=False)
        def sqdist_errors(samples, reencoded):
            per_vert_sqdist = (samples - reencoded).reshape(len(samples), len(samples[0])//3, 3)
            per_vert_sqdist = np.sum(np.abs(per_vert_sqdist)**2,axis=-1)
            return per_vert_sqdist
        
        def get_for_dim(pca_dim):
            U = components[:pca_dim].T

            explained_variance_ratio = 0

            def encode(samples):
                return samples @ U

            def decode(samples):
                return samples @ U.T

            per_vert_sqdist = sqdist_errors(samples, decode(encode(samples)))
            linf_list = np.max(np.sqrt(per_vert_sqdist), axis=-1)
            max_linf = np.max(linf_list)
            mean_linf = np.mean(linf_list)
            mean_l2 = np.mean(np.sum(per_vert_sqdist, axis=-1))
        
            return max_linf, mean_linf, mean_l2, U, explained_variance_ratio, encode, decode

        dim_list = list(reversed(range(1, len(samples[0]))))

        mem = mem_ if mem_ is not None else {}
        def bisect_left(n):
            lo = len(dim_list) - n # we're never going to work with a basis bigger than 300 right?
            hi = len(dim_list)
            while lo < hi:
                mid = (lo+hi)//2

                if dim_list[mid] not in mem:
                    mem[dim_list[mid]] = get_for_dim(dim_list[mid])
                print(dim_list[mid], f'max_linf={mem[dim_list[mid]][0]}, mean_linf={mem[dim_list[mid]][1]}, mean_l2={mem[dim_list[mid]][2]}')
                if mem[dim_list[mid]][0] < max_tol_error:
                    lo = mid+1
                else:
                    hi = mid
            return mem[dim_list[lo - 1]] if dim_list[lo - 1] in mem else get_for_dim(dim_list[lo - 1])

        max_linf, mean_linf, mean_l2, U, explained_variance_ratio, encode, decode = bisect_left(max_n_basis)
        n_U = U.shape[1]
        print("PCA basis of size", len(U[0]), " has max distance error of", max_linf, ', mean linf=', mean_linf, ', mean_l2=', mean_l2)
        
        return U, S[:n_U]
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    U, S = PCA(datas, args.max_tol_error, args.max_n_basis)
    print('PCA Done')
    print(f'Save PCA Basis to {args.output_dir}')
    igl.write_dmat(os.path.join(args.output_dir, "basis.dmat"), U, False)
    igl.write_dmat(os.path.join(args.output_dir, "eigenvalues.dmat"), S, False)
    

if __name__ == '__main__':
    main()
