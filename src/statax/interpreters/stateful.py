import logging
from inspect import signature
from typing import Any, Callable, Generator, Protocol, Union

import jax
import jax.tree_util as jtu
from jax._src import core, pjit, sharding_impls
from jax._src import linear_util as lu

from .. import caching, types, util
from .. import config as cfg
from .. import context as ctx
from ..util import StateError, StateMeta, strict_zip
from .common import generate_handler_store, jaxpr_vars_to_eq_vars, update_eqns

logger = logging.getLogger(__name__)

Sentinel = object
State = Union[dict[str, types.PyTree], Sentinel]
Args = Any
Kwargs = Any
Result = Any


class StatefulFn(Protocol):
  def __call__(
    self, *args: Args, state: State, **kwargs: Kwargs
  ) -> tuple[Result, State]: ...


_sentinel = object()

(
  get_primitive_handler,
  register_primitive_handler,
  _primitive_handlers,
) = generate_handler_store()


def _jit_handler(
  eq: core.JaxprEqn,
) -> tuple[core.JaxprEqn, tuple[StateMeta, ...]]:
  """Thread state through JIT

  Note does not work with donated args or shardings currently
  """
  cjpr = eq.params["jaxpr"]

  new_cjpr, inner_meta, translation = parse_jaxpr(cjpr)

  new_invars = jaxpr_vars_to_eq_vars(
    eq.invars, cjpr.jaxpr.invars, new_cjpr.jaxpr.invars, translation
  )

  new_outvars = jaxpr_vars_to_eq_vars(
    eq.outvars, cjpr.jaxpr.outvars, new_cjpr.jaxpr.outvars, translation
  )

  new_params = dict(eq.params)
  new_params["name"] += "_mod"
  new_params["jaxpr"] = new_cjpr
  new_params["out_shardings"] = tuple(
    sharding_impls.UnspecifiedValue() for _ in new_outvars
  )
  new_params["in_shardings"] = tuple(
    sharding_impls.UnspecifiedValue() for _ in new_invars
  )
  new_params["donated_invars"] = tuple(False for _ in new_invars)

  state_meta = inner_meta

  neweq = core.JaxprEqn(
    new_invars,
    new_outvars,
    eq.primitive,
    new_params,
    eq.effects,
    eq.source_info,
  )

  return neweq, state_meta


register_primitive_handler(pjit.pjit_p, _jit_handler)

# -------------------------------- Interpreter ------------------------------- #


def _thread_vars(new_eqns, new_inputs, new_outputs, metas: list[tuple[StateMeta, ...]]):
  """
  Thread state through equations.

  Uses the meta information on each equation to match common vars.


  """
  # All states seen in this jaxpr
  all_states = {v.name for k in metas for v in k}

  # output meta for this closed jaxpr
  out_metas: list[StateMeta] = []

  # Thread temporary vars correctly through all equations, for each state
  for state in all_states:
    state_metas: list[StateMeta] = []
    for eqn_metas in metas:
      meta_arr = tuple(filter(lambda v: v.name == state, eqn_metas))
      assert len(meta_arr) <= 1, "Multiple state variables with same name from one eqn."
      if len(meta_arr) == 0:
        state_metas.append(StateMeta())
      else:
        state_metas.append(meta_arr[0])

    seen = False
    didset = False
    current = None
    last_td = None
    out_meta = StateMeta(state)
    for i, meta in enumerate(state_metas):
      if meta.in_td is None and meta.out_td is None:
        continue
      if meta.in_idx != -1:  # getter
        if not seen:
          last_td = meta.in_td
          current = new_eqns[i].invars[meta.in_idx]
          new_inputs.append(current)
          out_meta = out_meta._replace(
            in_idx=len(new_inputs) - 1,
            in_td=meta.in_td,
            initfn=meta.initfn,
          )
        else:
          assert last_td == meta.in_td
          new_eqns[i].invars[meta.in_idx] = current
      if meta.out_idx != -1:  # setter
        # if not seen and thread_unused:
        last_td = meta.out_td
        didset = True
        current = new_eqns[i].outvars[meta.out_idx]
      seen = True
    if didset:
      new_outputs.append(current)
      out_meta = out_meta._replace(
        out_idx=len(new_outputs) - 1,
        out_td=last_td,
      )
    out_metas.append(out_meta)

  return new_eqns, new_inputs, new_outputs, tuple(out_metas)


def _get_used_vars(all_vars, new_eqns, new_outputs):
  """Identifies all Vars which are used in the new jaxpr."""
  used_dic = {k: False for k in all_vars if k.count != -1}
  for eq in new_eqns:
    for v in eq.invars:
      if isinstance(v, core.Var):
        used_dic[v] = True
  for v in new_outputs:
    # We can get literals in output
    if isinstance(v, core.Var):
      used_dic[v] = True
  return used_dic


def _update_vars(new_consts, new_inputs, new_outputs, new_eqns):
  """
  Generates new vars and a mapping from old vars.

  """
  all_vars: list[core.Var] = list(new_consts)
  all_vars.extend(new_inputs)
  for eq in new_eqns:
    all_vars.extend(eq.outvars)

  used_dic = _get_used_vars(all_vars, new_eqns, new_outputs)

  for k, v in tuple(used_dic.items()):
    if v is not True:
      used_dic.pop(k)
      all_vars.remove(k)

  assert all(used_dic.values())

  counter = 0
  main_lookup: dict[core.Var, core.Var] = {}
  for k in all_vars:
    if k not in main_lookup:
      if k.count == -1:
        main_lookup[k] = core.DropVar(aval=k.aval)
      else:
        main_lookup[k] = core.Var(count=counter, aval=k.aval, suffix=k.suffix)
      counter += 1

  return main_lookup


def parse_jaxpr(
  cjpr: core.ClosedJaxpr,
) -> tuple[core.ClosedJaxpr, tuple[StateMeta], dict[core.Var, core.Var]]:
  """Parses a ClosedJaxpr into a stateful ClosedJaxpr

  Returns
   - new jaxpr
   - a dictionary of meta information on how state is threaded
   - plus the lookup mapping vars in the original jaxpr to vars in the new jaxpr.

  """

  jpr = cjpr.jaxpr

  new_eqns, metas = update_eqns(jpr, get_primitive_handler)

  new_inputs = list(jpr.invars)
  new_outputs = list(jpr.outvars)
  new_eqns, new_inputs, new_outputs, out_metas = _thread_vars(
    new_eqns, new_inputs, new_outputs, metas
  )

  new_consts = list(jpr.constvars)
  main_lookup = _update_vars(new_consts, new_inputs, new_outputs, new_eqns)

  new_consts = [main_lookup[v] for v in new_consts if v in main_lookup]
  new_invars = [main_lookup[v] for v in new_inputs if v in main_lookup]
  new_outvars = [
    main_lookup[v] if isinstance(v, core.Var) and v in main_lookup else v
    for v in new_outputs
  ]
  for i, eq in enumerate(new_eqns):
    # handle literals
    ivs = [main_lookup[v] if isinstance(v, core.Var) else v for v in eq.invars]
    # handle dropped outputs
    ovs = [
      main_lookup.get(v, core.DropVar(v.aval)) if isinstance(v, core.Var) else v
      for v in eq.outvars
    ]
    new_eqns[i] = eq.replace(
      invars=ivs,
      outvars=ovs,
    )

  new_constvals = []
  for var, val in strict_zip(cjpr.jaxpr.constvars, cjpr.consts):
    if var in main_lookup:
      new_constvals.append(val)

  new_jpr = core.Jaxpr(new_consts, new_invars, new_outvars, new_eqns)
  new_cjpr = core.ClosedJaxpr(new_jpr, new_constvals)

  return new_cjpr, out_metas, main_lookup


@lu.transformation
def validate_state(
  init_input_state: dict[str, types.PyTree],
  in_states: dict[str, tuple[jtu.PyTreeDef, tuple[StateMeta]]],
  out_states: dict[str, tuple[jtu.PyTreeDef, tuple[StateMeta]]],
  *args: Any,
  state: dict[str, types.PyTree],
  **kwargs: Any,
) -> Generator[tuple[Any, dict], Any, None]:
  if state is _sentinel:
    state = {}
  if not isinstance(state, dict):
    raise StateError("State must be a dict[str, PyTree].")
  if not all((s in in_states or s in out_states) for s in state.keys()):
    s = tuple((s for s in state.keys() if s not in in_states and s not in out_states))
    raise StateError(f'Unknown state(s) "{s}" provided in input state')

  # copy dict, remove any states that are only outputs
  state_ = {k: v for k, v in state.items() if k in in_states}

  missing = (s for s in in_states.keys() if s not in state)

  state_.update({s: init_input_state[s] for s in missing})

  for v in jtu.tree_leaves(state_):
    assert not isinstance(v, core.AbstractValue)

  outkwargs = {"state": state_}
  outkwargs.update(kwargs)
  out = yield args, outkwargs
  yield out


@lu.transformation
def restructure(
  in_states,
  out_states,
  invars,
  results_td,
  *args: Any,
  state,
  **kwargs: Any,
) -> Generator[tuple[Any, dict], Any, None]:
  def_flat_args = jtu.tree_leaves((args, kwargs))

  state_flat_args = [None] * len(invars)
  state_flat_args[: len(def_flat_args)] = def_flat_args

  # Populate slots with state
  for s, (td, metas) in in_states.items():
    flat_ins, td_s = jtu.tree_flatten(state[s])
    assert td == td_s
    for i, m in enumerate(metas):
      taval = invars[m.in_idx].aval
      gaval = core.get_aval(flat_ins[i])
      assert taval.shape == gaval.shape, f"incorrect shape for state : {s}_{i}. \
          Got {gaval.shape}, expected {taval.shape}"
      assert taval.dtype == gaval.dtype, f"incorrect dtype for state : {s}_{i}. \
          Got {gaval.dtype}, expected {taval.dtype}"
      state_flat_args[m.in_idx] = flat_ins[i]

  outs = yield state_flat_args, {}

  flat_results = outs[: results_td.num_leaves]
  out_state = {
    s: jtu.tree_unflatten(
      td,
      [outs[m.out_idx] if m is not None else m for m in metas],
    )
    for s, (td, metas) in out_states.items()
  }

  results = jtu.tree_unflatten(results_td, flat_results)
  yield results, out_state


@util.cache_transformation(caching.default_cache.stateful)
def _transform(
  func,
  args,
  kwargs,
) -> tuple[Callable, dict[str, types.PyTree]]:
  """

  Returns
  -------
  Callable
    A new function generated from the modified jaxpr, with same input signature
    as original jaxpr, but additionally outputs dictionary of intermediates.
  """

  if cfg.default_config.debug:
    logger.info("binding")
  with ctx.Context(tracing=ctx.TracingCtx.STATEFUL):
    cjpr, shaped_results = jax.make_jaxpr(func, return_shape=True)(*args, **kwargs)
    newcjpr, state_meta, _ = parse_jaxpr(cjpr)

  spec_func_raw = lu.wrap_init(core.jaxpr_as_fun(newcjpr))

  batched_states: dict[str, dict[int, StateMeta]] = {}

  for v in state_meta:
    bn, i = util.to_batch_name(v.name)
    if bn not in batched_states:
      batched_states[bn] = {i: v}
    else:
      batched_states[bn][i] = v

  in_states: dict[str, tuple[jtu.PyTreeDef, tuple[StateMeta, ...]]] = {}
  out_states: dict[str, tuple[jtu.PyTreeDef, tuple[StateMeta, ...]]] = {}

  for s in batched_states:
    in_tds = [m.in_td for m in batched_states[s].values() if m.in_td is not None]
    if len(in_tds) > 0:
      assert all(t == in_tds[0] for t in in_tds)
      in_states[s] = (
        in_tds[0],
        tuple(batched_states[s][i] for i in range(in_tds[0].num_leaves)),
      )
    out_tds = [m.out_td for m in batched_states[s].values() if m.out_td is not None]
    if len(out_tds) > 0:
      assert all(t == out_tds[0] for t in out_tds)
      out_states[s] = (
        out_tds[0],
        tuple(batched_states[s][i] for i in range(out_tds[0].num_leaves)),
      )

  invars = newcjpr.jaxpr.invars

  results_td = jtu.tree_structure(shaped_results)

  init_input_state: dict[str, types.PyTree] = {
    s: jtu.tree_unflatten(
      td,
      [m.initfn() if m.initfn is not None else invars[m.in_idx].aval for m in metas],
    )
    for s, (td, metas) in in_states.items()
  }

  spec_func = restructure(
    spec_func_raw,
    in_states,
    out_states,
    invars,
    results_td,
  )

  spec_func = validate_state(
    spec_func,
    init_input_state,
    in_states,
    out_states,
  )

  return spec_func, init_input_state


@lu.transformation
def all_output(
  init_input_state,
  *args: Any,
  **kwargs: Any,
):
  in_state = kwargs["state"]
  result, state = yield args, kwargs

  if in_state is _sentinel:
    in_state = {}
  if not isinstance(in_state, dict):
    raise StateError("State must be a dict[str, PyTree].")

  state.update({k: v for k, v in in_state.items() if k not in state})
  state.update({k: v for k, v in init_input_state.items() if k not in state})
  yield result, state


def stateful_builder(
  func: Callable[[Args, Kwargs], Result],
  *,
  output_unchanged: bool = False,
) -> Callable[
  [Args, Kwargs],
  tuple[StatefulFn, State],
]:
  """
  Transforms a function to handle state.

  Generates a function which, when called with the same arguments as `func`,
  builds a transformed stateful version of `func` and the initial state
  dictionary.

  Parameters
  ----------
  func : Callable
    The function to be transformed.
  output_unchanged : bool, optional
    Whether to also output unchanged state values in the output state
    dictionary. Default False.

  Returns
  -------
  Callable[..., tuple[StatefulFn, State]]
    A builder for constructing the stateful function and initial state
    dictionary.

  """

  sig = signature(func)
  if "state" in sig.parameters:
    raise StateError("Function already has a state parameter.")

  def wrapped_func(*args: Args, **kwargs: Kwargs) -> tuple[StatefulFn, State]:
    specialised_func, init_input_state = _transform(func, args, kwargs)
    if output_unchanged:
      specialised_func = all_output(specialised_func, init_input_state)
    return specialised_func.call_wrapped, init_input_state

  return wrapped_func


def stateful(
  func: Callable[[Args, Kwargs], Result],
  *,
  output_unchanged: bool = False,
) -> StatefulFn:
  """
  Transforms a function to handle state.

  stateful :: (a -> b) -> ((a, Dict) -> (b, Dict))

  Parameters
  ----------
  func : Callable
    The function to be transformed.
  output_unchanged : bool, optional
    Whether to also output unchanged state values in the output state
    dictionary. Default False.

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

  builder = stateful_builder(func, output_unchanged=output_unchanged)

  def wrapped_func(
    *args: Args, state: State = _sentinel, **kwargs: Kwargs
  ) -> tuple[StatefulFn, State]:
    # build stateful function
    stateful_func, _ = builder(*args, **kwargs)
    return stateful_func(*args, state=state, **kwargs)

  return wrapped_func
