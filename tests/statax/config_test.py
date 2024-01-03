import statax as stx
from absl.testing import absltest


class TestConfig(absltest.TestCase):
  def test(self):
    assert stx.default_config.caching is True
    assert stx.default_config.debug is False
