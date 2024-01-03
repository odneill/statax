from enum import IntEnum, auto

from .config import Config, default_config


class TracingCtx(IntEnum):
  NONE = auto()
  PULL = auto()
  STATEFUL = auto()


_ctx_stack: list["Context"] = []


def get_ctx() -> "Context":
  """
  Get the current context.
  """
  if len(_ctx_stack) == 0:
    raise ValueError("No context on stack.")
  return _ctx_stack[-1]


class Context:
  """
  Store context for interpreting functions.
  """

  config: Config = default_config
  tracing: TracingCtx = TracingCtx.NONE

  def __init__(
    self,
    **kwargs,
  ) -> None:
    try:
      old_ctx = get_ctx()
    except ValueError:
      old_ctx = None
    for k in Context.__dict__:
      if k[:2] == "__":
        continue
      if k in kwargs:
        v = kwargs[k]
      elif old_ctx is not None:
        v = getattr(old_ctx, k)
      else:
        continue
      setattr(self, k, v)

  def __enter__(self):
    _ctx_stack.append(self)

  def __exit__(self, *_):
    _ctx_stack.pop(-1)


_ctx_stack.append(Context())


def get_config() -> Config:
  """Get the current config."""
  return get_ctx().config
