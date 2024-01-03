class Config:
  def __init__(self, **kwargs) -> None:
    self.reset()
    for k, v in kwargs.items():
      setattr(self, k, v)

  def __repr__(self) -> str:
    return "Config(" + ",".join([f"{k}={v}" for k, v in self.__dict__.items()]) + ")"

  def reset(self) -> None:
    self.caching = True
    self.debug = False


default_config = Config()
