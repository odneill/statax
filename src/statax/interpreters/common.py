from typing import Callable, Optional, Union

from jax._src import core

from ..util import StateMeta


def jaxpr_vars_to_eq_vars(
  oldeq_vars: list,
  oldjpr_vars: list[Union[core.Var, core.Literal]],
  newjpr_vars: list[Union[core.Var, core.Literal]],
  translation: Optional[dict[core.Var, core.Var]] = None,
):
  """Transform inner jaxpr vars to higher-order primitive vars

  With higher order primitives like jit, we have inner jaxprs which need to be
  transformed.
  The result is a new jaxpr with different vars. These need to be matched to
  the existing vars in the outer jaxpr eqns.

  If translation is None, then the vars are assumed to be the same, with only
  new vars potentially added.

  """
  neweq_vars = []
  if translation is None:

    def lookup(v):
      return v
  else:
    inv_translation = {v: k for k, v in translation.items()}

    def lookup(v):
      return inv_translation[v]

  for v in newjpr_vars:
    nv = lookup(v)
    if nv in oldjpr_vars:
      ind = oldjpr_vars.index(nv)
      neweq_vars.append(oldeq_vars[ind])
    else:
      neweq_vars.append(core.Var(count=0, aval=nv.aval, suffix=nv.suffix))
  return neweq_vars


Handler = Callable[[core.JaxprEqn], tuple[core.JaxprEqn, tuple[StateMeta, ...]]]


def generate_handler_store():
  """
  Generates a store for primitive handlers for a new interpreter.

  A handler takes a JaxprEqn, and returns a potentially modified version along
  with a tuple of StateMeta objects.

  If an equation needs extra input/ output then dummy core.Var's are added in
  these places, with the correct aval.

  The returned equations may be the original unchanged equation, a new equation
  """

  _primitive_handlers: dict[core.Primitive, Handler] = {}

  def _default_handler(
    eq: core.JaxprEqn,
  ) -> tuple[core.JaxprEqn, tuple[StateMeta, ...]]:
    return eq, ()

  def register_primitive_handler(primitive, handler):
    _primitive_handlers[primitive] = handler

  def get_primitive_handler(
    primitive: core.Primitive,
  ) -> Handler:
    return _primitive_handlers.get(primitive, _default_handler)

  return (
    get_primitive_handler,
    register_primitive_handler,
    _primitive_handlers,
  )


def update_eqns(
  jpr: core.Jaxpr, get_handler: Callable[[core.Primitive], Handler]
) -> tuple[list[core.JaxprEqn], list[tuple[StateMeta, ...]]]:
  # update stateful and higher order primitive equations
  new_eqns = []
  metas = []
  for eq in jpr.eqns:
    handler = get_handler(eq.primitive)
    new_eq, state_meta = handler(eq)
    metas.append(state_meta)
    new_eqns.append(new_eq)
  return new_eqns, metas
