import functools as ft
import logging
import sys
from typing import Any, Callable, NamedTuple

import jax.tree_util as jtu
from jax._src import core

from . import config as cfg

logger = logging.getLogger(__name__)


class StatefulWarning(Warning):
  pass


class StateError(Exception):
  pass


if sys.version_info >= (3, 10):
  strict_zip: Callable = ft.partial(zip, strict=True)
else:

  def strict_zip(*args):
    N = len(args[0])
    assert all(len(a) == N for a in args[1:])
    return zip(*args)


def to_batch_name(name: str) -> tuple[str, int]:
  splits = name.split("_")
  return "_".join(splits[:-1]), int(splits[-1])


def from_batch_name(name: str, index: int) -> str:
  return name + "_" + str(index)


def cache_transformation(cache: dict):
  """
  `inner` takes a function `func` and some arguments `args`. Assume `inner` is
  expensive to evaluate.

  This caching transformation replaces `inner` with a `cached_inner`, which only
  calls `inner` if the shape/ dtype of the args haven't been seen before with
  `func`.

  Otherwise `cached_inner` returns the cached version of the transformed function.
  """

  def decorator(inner):
    @ft.wraps(inner)
    def cached_inner(func, *args):
      flat_args, intree_def = jtu.tree_flatten(args)

      if cfg.default_config.caching:
        # Cache on __call__ to handle callable pytress with unhashable leaves
        # see https://github.com/python/mypy/issues/5079
        fnkey = func.__call__  # type: ignore
        if fnkey not in cache:
          cache[fnkey] = {}
        cache_ = cache[fnkey]
        key = hash((
          tuple(core.raise_to_shaped(core.get_aval(v)) for v in flat_args),
          intree_def,
        ))
        if key not in cache_:
          if cfg.default_config.debug:
            logger.info("cache miss")
          cache_[key] = inner(func, *args)
        wrapped_func = cache_[key]
      else:
        wrapped_func = inner(
          func,
          *args,
        )

      return wrapped_func

    return cached_inner

  return decorator


class StateMeta(NamedTuple):
  """Meta information for stateful functions.

  Informs threading of vars through stateful functions.
  """

  name: str = ""
  in_idx: int = -1
  out_idx: int = -1
  in_td: jtu.PyTreeDef = None
  out_td: jtu.PyTreeDef = None
  initfn: Any = None

  @property
  def batch_name(self):
    return to_batch_name(self.name)[0]

  @property
  def batch_index(self):
    return to_batch_name(self.name)[1]
