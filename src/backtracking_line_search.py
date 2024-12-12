from typing import NamedTuple, Union

import jax.numpy as jnp
import jax
from jax import lax
from jax import grad,value_and_grad


class _LineSearchState(NamedTuple):
  done: Union[bool, jax.Array]
  failed: Union[bool, jax.Array]
  i: Union[int, jax.Array]
  a_i1: Union[float, jax.Array]
  a_star: Union[float, jax.Array]
  f_star: Union[float, jax.Array]
  g_star: jax.Array


class _LineSearchResults(NamedTuple):
  """Results of line search.

  Parameters:
    failed: True if the strong Wolfe criteria were satisfied
    nit: integer number of iterations
    nfev: integer number of functions evaluations
    ngev: integer number of gradients evaluations
    k: integer number of iterations
    a_k: integer step size
    f_k: final function value
    g_k: final gradient value
    status: integer end status
  """
  failed: Union[bool, jax.Array]
  nit: Union[int, jax.Array]
  nfev: Union[int, jax.Array]
  ngev: Union[int, jax.Array]
  k: Union[int, jax.Array]
  a_k: Union[int, jax.Array]
  f_k: jax.Array
  g_k: jax.Array
  status: Union[bool, jax.Array]


def cond_fun(ls_result: _LineSearchResults):
    return (~ls_result.failed) & (~ls_result.done)


def line_search(f, xk, pk, old_fval=None, old_old_fval=None, gfk=None, c1=1e-4,
                c2=0.8, maxiter=20):
    if gfk is None:
        old_fval, gfk = value_and_grad(f)(xk)
    
    state = _LineSearchState(
      done=False,
      failed=False,
      # algorithm begins at 1 as per Wright and Nocedal, however Scipy has a
      # bug and starts at 0. See https://github.com/scipy/scipy/issues/12157
      i=1,
      a_i1=0.,
      a_star=0.,
      f_star=old_fval,
      g_star=gfk,
  )
    
    
    
    def body(state):
        a_i = jnp.where(state.i == 1, 1., state.a_i1 * c2)
        x = xk + a_i * pk
        fineshed = jnp.where(f(x) < f(xk) + c1 * a_i * jnp.dot(gfk, pk), True, False)
        state = state._replace(done=fineshed, i=state.i + 1, a_i1=a_i, a_star=a_i)
        
        return state
    
    state = lax.while_loop(lambda state: (~state.done) & (state.i <= maxiter) & (~state.failed),
                         body,
                         state)
    x = xk + state.a_star * pk
    f_v, g_v = value_and_grad(f)(x)
    results = _LineSearchResults(
      failed=state.failed | (~state.done),
      nit=state.i - 1,  # because iterations started at 1
      nfev=0,
      ngev=0,
      k=state.i,
      a_k=state.a_star,
      f_k=f_v,
      g_k=g_v,
      status=0,
    )
    return results
