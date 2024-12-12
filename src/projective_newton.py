import jax
import jax.numpy as jnp
import jax.scipy.optimize, jax.scipy.linalg
from jax import grad, jit, vmap, jacfwd, jacrev, Array, lax
from line_search import line_search
from typing import Any, Callable, Optional, Union, NamedTuple
import backtracking_line_search

class ProjectiveNewtonResults(NamedTuple):
  """Results from Projective Newton optimization

  Parameters:
    converged: True if minimization converged
    failed: True if non-zero status and not converged
    k: integer number of iterations of the main loop (optimisation steps)
    x_k: array containing the last argument value found during the search. If
      the search converged, then this value is the argmin of the objective
      function.
    f_k: array containing the value of the objective function at `x_k`. If the
      search converged, then this is the (local) minimum of the objective
      function.
    g_k: array containing the gradient of the objective function at `x_k`. If
      the search converged the l2-norm of this tensor should be below the
      tolerance.
    status: integer describing the status:
      0 = nominal  ,  1 = max iters reached,    4 = insufficient progress (ftol)
      5 = line search failed
    ls_status: integer describing the end status of the last line search
  """
  converged: Union[bool, Array]
  failed: Union[bool, Array]
  k: Union[int, Array]
  x_k: Array
  f_k: Array
  g_k: Array
  a_k: Array
  status: Union[int, Array]
  ls_status: Union[int, Array]
  ex_norm: Array


def minimize_projective_newton(
    fun: Callable,
    x0: jax.Array,
    maxiter: Optional[float] = None,
    norm=jnp.inf,
    project_eps = 1e-6,
    ftol: float = 2.220446049250313e-09,
    gtol: float = 1e-05,
    maxls: int = 20,
):
    d = len(x0)
    dtype = jnp.dtype(x0)

    # ensure there is at least one termination condition
    if maxiter is None:
        maxiter = d * 200

    # initial evaluation
    f_0, g_0 = jax.value_and_grad(fun)(x0)
    state_initial = ProjectiveNewtonResults(
        converged=False,
        failed=False,
        k=0,
        x_k=x0,
        f_k=f_0,
        g_k=g_0,
        a_k=0,
        status=0,
        ls_status=0,
        ex_norm=0,
    )

    def cond_fun(state: ProjectiveNewtonResults):
        return (~state.converged) & (~state.failed)

    def project(hessian):
        eigen_values, eigen_vectors = jnp.linalg.eigh(hessian)
        eigen_values = jnp.where(eigen_values < 0, 0, eigen_values)
        h = eigen_vectors @ jnp.diag(eigen_values) @ eigen_vectors.T
        h += project_eps * jnp.eye(hessian.shape[0])
        return h

    def body_fun(state: ProjectiveNewtonResults):
        # find search direction
        hessian = jacfwd(jacrev(fun))(state.x_k)
        p_k = -jax.scipy.linalg.solve(project(hessian), state.g_k, overwrite_a=True, assume_a='pos')
        
        ls_results = backtracking_line_search.line_search(
            f=fun,
            xk=state.x_k,
            pk=p_k,
            old_fval=state.f_k,
            gfk=state.g_k,
            maxiter=maxls,
        )
        # evaluate at next iterate
        s_k = ls_results.a_k.astype(p_k.dtype) * p_k
        # s_k = alpha * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = ls_results.f_k
        g_kp1 = ls_results.g_k

        # replacements for next iteration
        status = jnp.array(0)
        status = jnp.where(state.f_k - f_kp1 < ftol, 4, status) # saddle point reached
        status = jnp.where(state.k >= maxiter, 1, status)  # max newton iters reached
        status = jnp.where(ls_results.failed, 5, status) # line search failed

        e = jnp.linalg.norm(g_kp1, ord=norm)
        converged = e < gtol

        state = state._replace(
            converged=converged,
            failed=(status > 0) & (~converged),
            k=state.k + 1,
            x_k=x_kp1.astype(state.x_k.dtype),
            f_k=f_kp1.astype(state.f_k.dtype),
            g_k=g_kp1.astype(state.g_k.dtype),
            a_k=ls_results.a_k,
            status=jnp.where(converged, 0, status),
            ls_status=ls_results.status,
            ex_norm=e,
        )

        return state
    # return state_initial
    return lax.while_loop(cond_fun, body_fun, state_initial)
  
