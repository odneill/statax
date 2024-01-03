import statax.caching as caching
from absl.testing import absltest


class TestConfig(absltest.TestCase):
  def test(self):
    assert caching.default_cache.pull == {}
    assert caching.default_cache.stateful == {}
