from typing import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import statax as stx
from absl.testing import parameterized


def vmap(f: Callable) -> Callable:
  return lambda *args: jax.vmap(f)(*[jnp.repeat(a[None, :], 5, axis=0) for a in args])


class TestInters(parameterized.TestCase):
  def get_test_(self):
    def func(a, b):
      c = 2 * a
      c = stx.save_inter(c)
      c = b + c
      return jnp.sum(2 * c)

    ta = jnp.array([1.0, 2.0])
    tb = jnp.array([3.0, 4.0])

    ans = (4 * ta + 2 * tb).sum()
    grad = 0 * ta + 4.0
    return func, ta, tb, ans, grad

  def test_base(self):
    func, ta, tb, ans, _ = self.get_test_()

    assert func(ta, tb) == ans

    a, state = stx.pull(func)(ta, tb)
    assert a == ans
    assert jnp.all(state["state_0"] == 2 * ta)

  def test_jit(self):
    func, ta, tb, ans, _ = self.get_test_()

    assert jax.jit(func)(ta, tb) == ans

    a, state = stx.pull(jax.jit(func))(ta, tb)
    assert a == ans
    assert jnp.all(state["state_0"] == 2 * ta)

    a, state = jax.jit(stx.pull(func))(ta, tb)
    assert a == ans
    assert jnp.all(state["state_0"] == 2 * ta)

  def test_grad(self):
    func, ta, tb, ans, grad = self.get_test_()

    a, g = jax.value_and_grad(func)(ta, tb)  #
    assert a == ans
    assert jnp.all(g == grad)

    (a, g), state = stx.pull(jax.value_and_grad(func))(ta, tb)
    assert a == ans
    assert jnp.all(g == grad)
    assert jnp.all(state["state_0"] == 2 * ta)

    (a, state), g = jax.value_and_grad(stx.pull(func), has_aux=True)(ta, tb)
    assert a == ans
    assert jnp.all(g == grad)
    assert jnp.all(state["state_0"] == 2 * ta)

  def test_batch(self):
    func, ta, tb, ans, _ = self.get_test_()

    as_ = vmap(func)(ta, tb)
    assert jnp.all(jnp.array(as_) == ans)

    as_, state = stx.pull(vmap(func))(ta, tb)
    assert jnp.all(state["state_0"] == 2 * ta)
    assert jnp.all(jnp.array(as_) == ans)

    as_, state = vmap(stx.pull(func))(ta, tb)
    assert jnp.all(state["state_0"] == 2 * ta)
    assert jnp.all(jnp.array(as_) == ans)

  def test_pytree(self):
    def func(a):
      stx.save_inter(a, name="test")
      return a

    treein = (1.0, 2, {"as": 45}, [45, 7], jnp.array([12.0]))
    treeout = func(treein)
    assert all(
      jnp.all(a == b)
      for a, b in stx.util.strict_zip(
        jtu.tree_leaves(treein),
        jtu.tree_leaves(treeout),
      )
    )
    treeout, state = stx.pull(func)(treein)
    assert all(
      jnp.all(a == b)
      for a, b in stx.util.strict_zip(
        jtu.tree_leaves(treein),
        jtu.tree_leaves(treeout),
      )
    )
    assert all(
      jnp.all(a == b)
      for a, b in stx.util.strict_zip(
        jtu.tree_leaves(treein),
        jtu.tree_leaves(state["test_0"]),
      )
    )

  def test_kwargs(self):
    def func(b, a, *, c):
      stx.save_inter(a, name="a")
      return a + b * 3 * c

    pf = stx.pull(func)
    jf = jax.jit(func, static_argnames=("a",))
    pjf = stx.pull(jax.jit(func, static_argnames=("a",)))
    jpf = jax.jit(stx.pull(func), static_argnames=("a",))

    assert func(1, 2, c=3) == 2 + 1 * 3 * 3

    assert jf(1, 2, c=3) == 11
    assert jf(b=1, a=2, c=3) == 11
    assert jf(a=2, b=1, c=3) == 11
    assert jf(c=3, a=2, b=1) == 11

    assert pf(1, 2, c=3)[0] == 11
    assert pf(b=1, a=2, c=3)[0] == 11
    assert pf(a=2, b=1, c=3)[0] == 11
    assert pf(c=3, a=2, b=1)[0] == 11

    assert pjf(1, 2, c=3)[0] == 11
    assert pjf(b=1, a=2, c=3)[0] == 11
    assert pjf(a=2, b=1, c=3)[0] == 11
    assert pjf(c=3, a=2, b=1)[0] == 11

    assert jpf(1, 2, c=3)[0] == 11
    assert jpf(b=1, a=2, c=3)[0] == 11
    assert jpf(a=2, b=1, c=3)[0] == 11
    assert jpf(c=3, a=2, b=1)[0] == 11

  def test_caching(self):
    def func(b, a, *, c):
      stx.save_inter(a, name="a")
      return a + b * 3 * c

    stx.caching.default_cache.clear()

    pf = stx.pull(func)
    pjf = stx.pull(jax.jit(func, static_argnames=("a",)))
    jpf = jax.jit(stx.pull(func), static_argnames=("a",))

    for _ in range(3):
      assert func(1, 2, c=3) == 2 + 1 * 3 * 3

      assert pf(1, 2, c=3)[0] == 11
      assert pf(b=1, a=2, c=3)[0] == 11
      assert pf(a=2, b=1, c=3)[0] == 11
      assert pf(c=3, a=2, b=1)[0] == 11

      assert pjf(1, 2, c=3)[0] == 11
      assert pjf(b=1, a=2, c=3)[0] == 11
      assert pjf(a=2, b=1, c=3)[0] == 11
      assert pjf(c=3, a=2, b=1)[0] == 11

      assert jpf(1, 2, c=3)[0] == 11
      assert jpf(b=1, a=2, c=3)[0] == 11
      assert jpf(a=2, b=1, c=3)[0] == 11
      assert jpf(c=3, a=2, b=1)[0] == 11

    cache_size = [len(a) for a in stx.caching.default_cache.pull.values()]
    print(f"{cache_size=}")  # only shows on fail
    # func and PjitFunction, each
    assert all(a == 2 for a in cache_size)

  def test_timing(self):
    def func(a):
      stx.save_inter(a, name="a")
      return a

    treein = (1.0, 2, {"as": 45}, [45, 7], jnp.array([12.0]))

    pf = stx.pull(func)
    jf = jax.jit((func))
    jpf = jax.jit(stx.pull(func))
    pjf = stx.pull(jax.jit(func))

    from timeit import Timer

    stx.default_config.caching = True

    cache_timings = (
      (lambda a: (a[0], a[1] / a[0]))(Timer(lambda: pf(treein)).autorange()),
      (lambda a: (a[0], a[1] / a[0]))(Timer(lambda: jf(treein)).autorange()),
      (lambda a: (a[0], a[1] / a[0]))(Timer(lambda: jpf(treein)).autorange()),
      (lambda a: (a[0], a[1] / a[0]))(Timer(lambda: pjf(treein)).autorange()),
    )

    stx.default_config.caching = False

    nocache_timings = (
      (lambda a: (a[0], a[1] / a[0]))(Timer(lambda: pf(treein)).autorange()),
      (lambda a: (a[0], a[1] / a[0]))(Timer(lambda: jf(treein)).autorange()),
      (lambda a: (a[0], a[1] / a[0]))(Timer(lambda: jpf(treein)).autorange()),
      (lambda a: (a[0], a[1] / a[0]))(Timer(lambda: pjf(treein)).autorange()),
    )

    print(f"{cache_timings=}")  # only shows on fail
    print(f"{nocache_timings=}")  # only shows on fail

    assert all(
      a[1] < 5 * b[1] for a, b in stx.util.strict_zip(cache_timings, nocache_timings)
    )

  def test_same(self):
    def f(x):
      y = 4 * jnp.ones(())
      z = x * y
      z = stx.save_inter(z, name="z")
      return z

    a, state = stx.pull(jax.jit(f))(1.0)
    assert a == 4.0
    assert state["z_0"] == 4.0
    assert len(state) == 1
