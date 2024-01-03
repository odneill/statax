import logging
from typing import Any, Callable, Generator

import jax
import jax.tree_util as jtu
from jax._src import core, pjit, sharding_impls
from jax._src import linear_util as lu

from .. import caching, util
from .. import config as cfg
from .. import context as ctx
from ..util import StateMeta, to_batch_name
from .common import generate_handler_store, jaxpr_vars_to_eq_vars, update_eqns

logger = logging.getLogger(__name__)

(
  get_primitive_handler,
  register_primitive_handler,
  _primitive_handlers,
) = generate_handler_store()


def _jit_handler(
  eq: core.JaxprEqn,
) -> tuple[core.JaxprEqn, tuple[StateMeta, ...]]:
  """Thread out of JIT"""
  cjpr = eq.params["jaxpr"]

  new_cjpr, inner_meta = parse_jaxpr(cjpr)

  new_outvars = jaxpr_vars_to_eq_vars(
    eq.outvars, cjpr.jaxpr.outvars, new_cjpr.jaxpr.outvars
  )

  new_params = dict(eq.params)
  new_params["name"] += "_mod"
  new_params["jaxpr"] = new_cjpr
  new_params["out_shardings"] = tuple(
    sharding_impls.UnspecifiedValue() for _ in new_outvars
  )

  state_meta = inner_meta

  neweq = core.JaxprEqn(
    eq.invars,
    new_outvars,
    eq.primitive,
    new_params,
    eq.effects,
    eq.source_info,
  )

  return neweq, state_meta


register_primitive_handler(pjit.pjit_p, _jit_handler)


# -------------------------------- Interpreter ------------------------------- #


def parse_jaxpr(
  cjpr: core.ClosedJaxpr,
) -> tuple[core.ClosedJaxpr, tuple[StateMeta, ...]]:
  """
  Transforms jaxpr in order to extract marked intermediates.

  Parameters
  ----------
  cjpr : core.ClosedJaxpr
    The ClosedJaxpr object to be parsed.

  Returns
  -------
  core.ClosedJaxpr
    The parsed ClosedJaxpr object.
  tuple[StateMeta, ...]
    A set of names of metas detailing the intermediates present.
  """

  jpr = cjpr.jaxpr

  new_eqns, metas = update_eqns(jpr, get_primitive_handler)

  outvars = jpr.outvars
  N = len(outvars)
  states = []
  out_metas = []

  j = 0
  for i, m in enumerate(metas):
    for v in m:
      states.append(new_eqns[i].outvars[v.out_idx])
      out_metas.append(v._replace(out_idx=j + N))
      j += 1

  new_outvars = [*outvars, *states]

  new_jpr = core.Jaxpr(jpr.constvars, jpr.invars, new_outvars, new_eqns)
  new_cjpr = core.ClosedJaxpr(new_jpr, cjpr.consts)

  return new_cjpr, tuple(out_metas)


@lu.transformation
def restructure(
  outtree: jtu.PyTreeDef,
  safe_state_names: list[str],
  *args: Any,
  **kwargs: Any,
) -> Generator[tuple[Any, dict], Any, None]:
  flat_args = jtu.tree_leaves((args, kwargs))
  outs = yield flat_args, {}
  outs = jtu.tree_unflatten(outtree, outs)
  outs = (
    outs[0],
    dict(zip(safe_state_names, outs[1])),
  )
  yield outs


def get_state_structure(
  state_meta: tuple[StateMeta, ...],
) -> tuple[list[str], list[jtu.PyTreeDef]]:
  """
  Generates structural information for reconstructing outputs from flat vars.
  """
  name_counts: dict[str, int] = {}
  safe_state_names = []
  group_structs = []
  for m in state_meta:
    bn, i = to_batch_name(m.name)
    if i > 0:
      # Already seen the group, skip the rest
      continue
    if bn in name_counts:
      name_counts[bn] += 1
    else:
      name_counts[bn] = 0
    safe_state_names.append(bn + "_" + str(name_counts[bn]))
    group_structs.append(jtu.tree_unflatten(m.out_td, [0.0] * m.out_td.num_leaves))
  return safe_state_names, group_structs


@util.cache_transformation(caching.default_cache.pull)
def _transform(func, args, kwargs) -> lu.WrappedFun:
  """

  Takes a function and example (args, kwargs) for that function

  Returns a transformed version of that function, specialised to the structure
  of the example args.

  Returns
  -------
  Callable
    A new function generated from the modified jaxpr, with same input signature
    as original jaxpr, but additionally outputs dictionary of intermediates.
  """

  if cfg.default_config.debug:
    logger.info("binding")
  with ctx.Context(tracing=ctx.TracingCtx.PULL):
    cjpr, outtree_val = jax.make_jaxpr(func, return_shape=True)(*args, **kwargs)
    newcjpr, state_meta = parse_jaxpr(cjpr)

  spec_func_raw = lu.wrap_init(core.jaxpr_as_fun(newcjpr))

  safe_state_names, group_structs = get_state_structure(state_meta)

  outtree = jtu.tree_structure((outtree_val, group_structs))

  # Wrap the raw interpreted function with tree flattening and unflattening
  spec_func = restructure(
    spec_func_raw,
    outtree,
    safe_state_names,
  )

  return spec_func


def pull(func: Callable) -> Callable:
  """
  Transforms a function to extract intermediates.

  pull :: (a -> b) -> (a -> (b, Dict))

  Parameters
  ----------
  func : Callable
    The function to be transformed.

  Returns
  -------
  Callable
    The transformed function that returns extracted intermediates along with
    original output.

  Notes
  -----
  This function takes a function as input and returns a new function that
  extracts intermediates.
  The returned function can be called with the same arguments as the original
  function, but it also
  returns a dictionary containing the intermediate values.

  Examples
  --------
  >>> def add(a, b):
  ...   stx.save_inters(2*a, name="a")
  ...   return a + b
  ...
  >>> add_with_intermediates = stx.pull(add)
  >>> result, intermediates = add_with_intermediates(2, 3)
  >>> print(result)
  5
  >>> print(intermediates)
  {'a_0': 4}
  """

  def wrapped_func(*args: Any, **kwargs: Any) -> tuple[Any, dict]:
    specialised_func = _transform(func, args, kwargs)
    return specialised_func.call_wrapped(*args, **kwargs)

  return wrapped_func
