import functools as ft
import logging
import warnings
from enum import IntEnum, auto
from typing import Any, Callable, Optional

import jax.tree_util as jtu
from jax._src import core
from jax.interpreters import ad, batching, mlir

from . import config as cfg
from . import context as ctx
from . import types
from .interpreters.inters import (
  register_primitive_handler as pull_register_handler,
)
from .interpreters.stateful import (
  register_primitive_handler as stateful_register_handler,
)
from .util import StateError, StatefulWarning, StateMeta, from_batch_name

logger = logging.getLogger(__name__)


class StatefulOp(IntEnum):
  SAVE = auto()
  SET = auto()
  GET = auto()


# ---------------------------------- Inters ---------------------------------- #

statevar_p = core.Primitive("statevar")


def _statevar(
  op: StatefulOp,
  x: types.PyTree,
  name: Optional[str] = None,
  initfn: Optional[Callable[[], types.PyTree]] = None,
) -> types.PyTree:
  """ """

  if name is None:
    if op != StatefulOp.SAVE:
      raise StateError("Name must be provided for stateful operations.")
    name = "state"
  in_leaves, treedef = jtu.tree_flatten(x)

  if initfn is not None:

    def wrapped_initfn(i):
      val = jtu.tree_leaves(initfn())[i]
      assert not isinstance(val, (core.Tracer, core.AbstractValue))
      return val

  out_leaves = [
    statevar_p.bind(
      leaf,
      name=from_batch_name(name, i),
      td=treedef,
      disable=False,
      op=op,
      initfn=ft.partial(wrapped_initfn, i) if initfn is not None else None,
    )
    for i, leaf in enumerate(in_leaves)
  ]
  return jtu.tree_unflatten(treedef, out_leaves)


def _statevar_impl(
  x: Any,
  name: str,
  td: jtu.PyTreeDef,
  disable: bool,
  op: StatefulOp,
  initfn: Callable[[], Any],
):
  if not disable:
    if op == StatefulOp.SET:
      if ctx.get_ctx().tracing != ctx.TracingCtx.STATEFUL:
        warnings.warn(
          "set_state is a no-op until transformed into a stateful function",
          StatefulWarning,
        )
    elif op == StatefulOp.GET:
      if ctx.get_ctx().tracing != ctx.TracingCtx.STATEFUL:
        warnings.warn(
          "get_state relies only on initialiser until transformed into a \
          stateful function",
          StatefulWarning,
        )
  return x


def _statevar_abstract_eval(x, **kwargs):
  return x


def _statevar_lowering(ctx, xc, **kwargs):
  return [xc]


def _statevar_batch(vector_arg_values, batch_axes, **kwargs):
  res = statevar_p.bind(*vector_arg_values, **kwargs)
  return res, batch_axes[0]


def _statevar_handler(
  eq: core.JaxprEqn,
) -> tuple[core.JaxprEqn, tuple[StateMeta, ...]]:
  op = eq.params["op"]
  if eq.params["disable"]:
    return eq, ()
  if cfg.default_config.debug:
    logger.info(f"hit statevar : {op}")
  name = eq.params["name"]
  ov = eq.outvars[0]
  params = dict(eq.params)
  # Need to disable operation - multiple stateful transformations will break things
  params["disable"] = True
  if op in (StatefulOp.SET, StatefulOp.SAVE):
    # input is -1, saying we don't need to change the input at all
    # output is 0, saying we need to use the outvar in index 0 as the output
    meta = StateMeta(name, -1, 0, None, eq.params["td"], eq.params["initfn"])
    if ov.count == -1:
      outvars = [core.Var(count=0, aval=ov.aval, suffix=ov.suffix)]
    else:
      outvars = eq.outvars
    invars = eq.invars
  elif op == StatefulOp.GET:
    # input is zero, saying we should insert state in arg position 0
    # output is -1, saying no modification to output is needed
    meta = StateMeta(name, 0, -1, eq.params["td"], None, eq.params["initfn"])
    invars = [core.Var(count=0, aval=ov.aval, suffix=ov.suffix)]
    outvars = eq.outvars

  eq = eq.replace(
    invars=invars,
    outvars=outvars,
    params=params,
  )
  return eq, (meta,)


def _statevar_stateful(
  eq: core.JaxprEqn,
) -> tuple[core.JaxprEqn, tuple[StateMeta, ...]]:
  if ctx.get_ctx().tracing != ctx.TracingCtx.STATEFUL:
    return eq, ()
  return _statevar_handler(eq)


def _statevar_pull(
  eq: core.JaxprEqn,
) -> tuple[core.JaxprEqn, tuple[StateMeta, ...]]:
  if ctx.get_ctx().tracing != ctx.TracingCtx.PULL:
    return eq, ()
  return _statevar_handler(eq)


statevar_p.def_impl(_statevar_impl)
statevar_p.def_abstract_eval(_statevar_abstract_eval)
mlir.register_lowering(statevar_p, _statevar_lowering, platform="cpu")
mlir.register_lowering(statevar_p, _statevar_lowering, platform="gpu")

ad.deflinear(statevar_p, lambda t, *args, **kwargs: [t])
batching.primitive_batchers[statevar_p] = _statevar_batch

stateful_register_handler(statevar_p, _statevar_stateful)
pull_register_handler(statevar_p, _statevar_pull)


def save_inter(
  x: types.PyTree,
  *,
  name: Optional[str] = None,
) -> types.PyTree:
  return _statevar(StatefulOp.SAVE, x, name=name)


def set_state(
  x: types.PyTree,
  *,
  name: str,
) -> types.PyTree:
  return _statevar(StatefulOp.SET, x, name=name)


def get_state(
  *,
  name: str,
  init: Callable[[], types.PyTree],
) -> types.PyTree:
  """ """

  try:
    val = init()
  except TypeError as e:
    raise StateError("Bad initialiser provided.") from e

  return _statevar(StatefulOp.GET, val, name=name, initfn=init)
