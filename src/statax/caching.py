class _Cache:
  """Cache for transformed functions."""

  pull: dict
  stateful: dict

  def __init__(self) -> None:
    self.clear()

  def clear(self) -> None:
    self.pull = {}
    self.stateful = {}

  def __repr__(self) -> str:
    return "_Cache(" + ",".join([f"{k}={v}" for k, v in self.__dict__.items()]) + ")"


default_cache = _Cache()
