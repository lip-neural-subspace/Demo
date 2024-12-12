from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

def generate_fixed_entry_data(fixed_mask, initial_values):
    '''
    Preprocess to construct data to efficiently apply fixed values (e.g. pinned vertices, boundary conditions) to a vector.

    Suppose we want a vector of length N, with F fixed entries.
        - `fixed_mask` is a length-N boolean array, where `True` entries are fixed
        - `initial_values` is a length-N float array giving the fixed values. Elements corresponding to non-fixed entries are ignored.
    '''
    
    fixed_inds = jnp.nonzero(fixed_mask)[0]
    unfixed_inds = jnp.nonzero(~fixed_mask)[0]
    fixed_values = initial_values[fixed_mask]
    unfixed_values = initial_values[~fixed_mask]

    return fixed_inds, unfixed_inds, fixed_values, unfixed_values

def apply_fixed_entries(fixed_inds, unfixed_inds, fixed_values, unfixed_values):
    '''
    Applies fixed values to a vector, using the indexing arrays generated from generate_fixed_entry_data(), plus a vector of all the un-fixed values.

    Passing fixed_values or unfixed_values as a scalar will also work.
    '''

    out = jnp.zeros(fixed_inds.shape[0] + unfixed_inds.shape[0])
    out = out.at[fixed_inds].set(fixed_values, indices_are_sorted=True, unique_indices=True)
    out = out.at[unfixed_inds].set(unfixed_values, indices_are_sorted=True, unique_indices=True)

    return out

# mc
def mc_apply_fixed_entries(fixed_inds, unfixed_inds, fixed_values, unfixed_values):
    '''
    Applies fixed values to a vector, using the indexing arrays generated from generate_fixed_entry_data(), plus a vector of all the un-fixed values.

    Passing fixed_values or unfixed_values as a scalar will also work.
    '''

    out = jnp.zeros((fixed_inds.shape[0] + unfixed_inds.shape[0], 4))
    out = out.at[fixed_inds].set(fixed_values, indices_are_sorted=True, unique_indices=True)
    out = out.at[unfixed_inds].set(unfixed_values, indices_are_sorted=True, unique_indices=True)

    return out

# cubature 
# @partial(jax.jit,
#          static_argnames=['cubature_inds'])
def get_cubature_elements(F, cubature_inds):
    return F[cubature_inds]

# get cubature E and pca basis
def sub_pca_and_e(E, all_fixed_old_i, all_unfixed_old_i, cubature_inds, basis_p, U, n_dof):
    sub_e_old_v = E[cubature_inds, :]
    all_fixed_old_i_set = set(all_fixed_old_i.tolist())
    sub_unfixed_old_i_set = set()
    for e_old_v in sub_e_old_v:
        for old_v in e_old_v:
            for old_i in range(old_v * 3, old_v * 3 + 3):
                if old_i not in all_fixed_old_i_set:
                    sub_unfixed_old_i_set.add(old_i)
    sub_unfixed_old_i = list(sub_unfixed_old_i_set)
    sub_unfixed_old_i.sort()
    
    sub_old_i = jnp.concatenate((jnp.array(sub_unfixed_old_i), all_fixed_old_i))
    sub_old_i_to_new_i = {}
    for new_i, old_i in enumerate(sub_old_i):
        sub_old_i_to_new_i[int(old_i)] = new_i
    sub_e_new_v = np.zeros_like(sub_e_old_v)
    for i, e_old_v in enumerate(sub_e_old_v):
        for j, old_v in enumerate(e_old_v):
            sub_e_new_v[i, j] = sub_old_i_to_new_i[int(old_v) * 3] // 3
    all_unfixed_old_i_to_U_i = np.zeros(n_dof, dtype=int)
    for U_i, old_i in enumerate(all_unfixed_old_i):
        all_unfixed_old_i_to_U_i[old_i] = U_i
    sub_U = U[all_unfixed_old_i_to_U_i[sub_unfixed_old_i], :]
    sub_basis_p = basis_p[all_unfixed_old_i_to_U_i[sub_unfixed_old_i]]
    return sub_U, sub_e_new_v, sub_basis_p
