import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
import statax as stx
from absl.testing import parameterized


def vmap(f):
  def mapper(*a, **k):
    def _f(_):
      return (a, k)

    args = jax.vmap(_f)(jnp.arange(5))
    return jax.vmap(f)(*args[0], **args[1])

  return mapper


class TestStateful(parameterized.TestCase):
  def get_test_(self):
    def func(a, b):
      c = 2 * a
      d = stx.get_state(name="c", init=lambda: 0.0)
      c = b + c + d
      return jnp.sum(2 * c)

    ta = jnp.array([1.0, 2.0])
    tb = jnp.array([3.0, 4.0])
    state_ = {"c": 2.0}

    ans_ini = (4 * ta + 2 * tb).sum()
    ans = (4 * ta + 2 * tb + 2 * state_["c"]).sum()
    grada = 0 * ta + 4.0
    gradb = 0 * tb + 2.0
    gradc = (0 * tb + 2.0).sum()
    return func, ta, tb, state_, ans_ini, ans, (grada, gradb, gradc)

  def test_base(self):
    func, ta, tb, state_, ans_ini, ans, _ = self.get_test_()
    with pytest.warns(stx.StatefulWarning):
      assert func(ta, tb) == ans_ini

    a, state = stx.stateful(func)(ta, tb, state=state_)
    assert a == ans
    assert state == {}

  def test_jit(self):
    func, ta, tb, state_, ans_ini, ans, _ = self.get_test_()

    assert jax.jit(func)(ta, tb) == ans_ini

    a, state = stx.stateful(jax.jit(func))(ta, tb, state=state_)
    assert a == ans
    assert state == {}

    a, state = jax.jit(stx.stateful(func))(ta, tb, state=state_)
    assert a == ans
    assert state == {}

  def test_grad(self):
    func, ta, tb, state_, ans_ini, ans, (grada, gradb, gradc) = self.get_test_()
    with pytest.warns(stx.StatefulWarning):
      a, g = jax.value_and_grad(func)(ta, tb)
    assert a == ans_ini
    assert jnp.all(g == grada)

    (a, g), state = stx.stateful(jax.value_and_grad(func))(ta, tb, state=state_)
    assert a == ans
    assert jnp.all(g == grada)
    assert state == {}

    (a, state), g = jax.value_and_grad(stx.stateful(func), has_aux=True)(
      ta, tb, state=state_
    )
    assert a == ans
    assert jnp.all(g == grada)
    assert state == {}

    (a, state), g = jax.value_and_grad(
      stx.stateful(func), argnums=(0, 1), has_aux=True
    )(ta, tb, state=state_)
    assert a == ans
    assert jnp.all(g[0] == grada)
    assert jnp.all(g[1] == gradb)
    assert state == {}

    def lf(a, b, c):
      return stx.stateful(func)(a, b, state={"c": c})

    (a, state), g = jax.value_and_grad(lf, argnums=(0, 1, 2), has_aux=True)(
      ta, tb, state_["c"]
    )
    assert a == ans
    assert jnp.all(g[0] == grada)
    assert jnp.all(g[1] == gradb)
    assert jnp.all(g[2] == gradc)
    assert state == {}

  def test_batch(self):
    func, ta, tb, state_, ans_ini, ans, _ = self.get_test_()

    with pytest.warns(stx.StatefulWarning):
      as_ = vmap(func)(ta, tb)
    assert jnp.all(jnp.array(as_) == ans_ini)

    as_, state = stx.stateful(vmap(func))(ta, tb, state=state_)
    assert state == {}
    assert jnp.all(jnp.array(as_) == ans)

    as_, state = vmap(stx.stateful(func))(ta, tb, state=state_)
    assert state == {}
    assert jnp.all(jnp.array(as_) == ans)

  def test_pytree(self):
    def func(a):
      stx.set_state(a, name="test")
      return a

    treein = (1.0, 2, {"as": 45}, [45, 7], jnp.array([12.0]))
    with pytest.warns(stx.StatefulWarning):
      treeout = func(treein)
    assert all(
      jnp.all(a == b)
      for a, b in stx.util.strict_zip(
        jtu.tree_leaves(treein),
        jtu.tree_leaves(treeout),
      )
    )
    treeout, state = stx.stateful(func)(treein)
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
        jtu.tree_leaves(state["test"]),
      )
    )

  def test_pytree2(self):
    treea0 = (0.0, 0, {"as": 0}, [0, 0], jnp.array([0.0]))  # noqa: F841
    treea = (1.0, 2, {"as": 45}, [45, 7], jnp.array([12.0]))
    treeb0 = (jnp.array([0.0, 0.0, 0.0]), {"bs": 0, "cs": None}, [0, 0.0])
    treeb = (jnp.array([1.0, 2.0, 3.0]), {"bs": 5, "cs": None}, [93, 3.14159])

    def func(a):
      b = stx.get_state(name="b", init=lambda: treeb0)
      stx.set_state(a, name="a")
      return b

    with pytest.warns(stx.StatefulWarning):
      treeout = func(treea)
    assert all(
      jnp.all(a == b)
      for a, b in stx.util.strict_zip(
        jtu.tree_leaves(treeb0),
        jtu.tree_leaves(treeout),
      )
    )
    treeout, state = stx.stateful(func)(treea, state={"b": treeb})
    assert all(
      jnp.all(a == b)
      for a, b in stx.util.strict_zip(
        jtu.tree_leaves(treeb),
        jtu.tree_leaves(treeout),
      )
    )
    assert all(
      jnp.all(a == b)
      for a, b in stx.util.strict_zip(
        jtu.tree_leaves(treea),
        jtu.tree_leaves(state["a"]),
      )
    )

  def test_same(self):
    def f(x):
      y = stx.get_state(name="y", init=lambda: 4 * jnp.ones(()))
      z = x * y
      z = stx.set_state(z, name="z")
      return z

    stx.stateful(jax.jit(f))(1.0, state={"y": 3.0})

  def test_auto_init(self):
    def f(x):
      y = stx.get_state(name="y", init=lambda: 4 * jnp.ones(()))
      z = x * y
      z = stx.set_state(z, name="z")
      return z

    stx.stateful(f)(1.0)

  def test_unchanged(self):
    def f(x):
      y = stx.get_state(name="y", init=lambda: 4 * jnp.ones(()))
      z = x * y
      z = stx.set_state(z, name="z")
      return z

    val, state = stx.stateful(f)(1.0)
    assert val == 4.0
    assert state["z"] == 4.0
    assert len(state) == 1

    val, state = stx.stateful(f, output_unchanged=True)(2.0)

    assert val == 8.0
    assert state["z"] == 8.0
    assert state["y"] == 4.0
    assert len(state) == 2

    # y should be the input value, not the initial value, even though it's not
    # set in the function
    val, state = stx.stateful(f, output_unchanged=True)(2.0, state={"y": 7.0})

    assert val == 14.0
    assert state["z"] == 14.0
    assert state["y"] == 7.0
    assert len(state) == 2

  def test_unused_state(self):
    def f(x):
      y = stx.get_state(name="y", init=lambda: 4 * jnp.ones(()))
      z = x * y
      z = stx.set_state(z, name="z")
      return z

    val, state = stx.stateful(f)(1.0, state={"z": 3.0})

    assert val == 4.0
    assert state["z"] == 4.0
    assert len(state) == 1
