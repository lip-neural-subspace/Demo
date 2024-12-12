from functools import partial

import numpy as np
import scipy.optimize
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import lbfgs
import projective_newton
methods = [
    'L-BFGS',
    'Projective-Newton',
]

# solve
# returns a tuple of: (result, solver_info)
# where solver_info is a dictionary of solver-defined statistics
def minimize_nonlinear(func, init_guess, method):
    print('compiling def minimize_nonlinear(func, init_guess, method)...')

    if method == 'L-BFGS':
        res = lbfgs.minimize_lbfgs(func, init_guess, maxiter=8192, maxls=128, gtol=1e-5)
        return (res.x_k, {
            "success": ~res.failed,
            "scipy_optimize_status": res.status,
            "n_iter": res.k,
            "norm": res.ex_norm,
        })
    elif method == 'Projective-Newton':
        res = projective_newton.minimize_projective_newton(func, init_guess, maxiter=8192, maxls=128)
        return (res.x_k, {
            "success": ~res.failed,
            "scipy_optimize_status": res.status,
            "n_iter": res.k,
            "norm": res.ex_norm,
        })
    else:
        raise ValueError("unrecognized method")


def print_solver_info(info):
    print("  solver info:")
    for k in info:
        val = info[k]
        if k in ['success']:
            val = bool(val)
        if k in ['n_iter']:
            val = int(val)
        if k in ['scipy_optimize_status']:
            val = {
                0: "converged (nominal)",
                1: "max BFGS iters reached",
                3: "zoom failed",
                4: "saddle point reached",
                5: "line search failed",
                -1: "undefined"
            }[int(val)]
        info[k] = val
     
        print(f"   {k:>30}: {val}")
