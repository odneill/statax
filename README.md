# Statax

Statax is an experimental custom interpreter for [JAX](https://github.com/google/jax), allowing for injection/ extraction of state from otherwise pure JAX functions.

### Usage

We provide two new function transformations, `pull` and `stateful`:
- `pull` allows for extraction of intermediate values from a function,
- `stateful` allows for full threading of state through a function.

The transformations act as follows:
- pull :: (a -> b) -> (a -> (b, dict))
- stateful :: (a -> b) -> ((a, dict) -> (b, dict))

Intermediate values are marked for extraction via `pull` by passing them to `save_inter`, which otherwise acts as a no-op.

State values handled by `stateful` are accessed via `get_state` and `set_state`.
- `get_state` takes a name and a callable which initialises the state if it is not already present, and returns the value of the state. 
- `set_state` takes a name and a value, and returns the value.

For both `pull` and `stateful`, allowed values are any PyTree of valid jaxtypes.

Both transformations are compatible and composable with familiar JAX transformations i.e. `jit`, `grad`, `vmap` etc.

```python
from functools import partial

import jax
import jax.numpy as jnp
import statax as stx

# ---------------------------- Intermediates ----------------------------

def f(x):
  y = x + 1
  stx.save_inter(y, name="y")
  z = 2 * y
  return z

f(2.0)
# outputs 6.0

stx.pull(f)(2.0)
# outputs (Array(6., dtype=float32, weak_type=True), {'y_0': Array(3., dtype=float32, weak_type=True)})

# ------------------------------ Stateful -------------------------------

def g(x):
  y = stx.get_state(name="y", init=partial(jnp.ones, ()))
  z = x * y
  stx.set_state(z, name="y")
  return z + 5.0

# State comes entirely from initialiser, set_state has no effect
g(2.0) 
# outputs Array(7., dtype=float32)

# State comes from initialiser, set_state is output with original results
stx.stateful(g)(2.0) 
# outputs (Array(7., dtype=float32), {'y': Array(2., dtype=float32)})

# State is input in state dictionary, and modified state is returned
stx.stateful(g)(2.0, state={"y": 5.0}) 
# outputs (Array(15., dtype=float32), {'y': Array(10., dtype=float32)})
```

Note: This is an experimental library, and there will be bugs/ missed cases. In particular, support for threading through some higher-order `lax` primitives such as `cond`, `scan`, `while` is still missing, although in most of these cases there are workarounds such as capture via closure.

Note: All functionality of `pull` is achievable via `stateful`. `pull` was developed first, and is provided as a convenience, as a slightly lighter interpreter than `stateful`.

### Design

There exist many libraries/ patterns which allow handling state in JAX.

Manually threading state alongside function arguments and return values is conceptually the simplest approach, however this typically requires significant modification to exiting state-less code, especially for deeply nested functions.

JAX itself has an experimental `run_state` API, however this is not currently stable, and still requires threading references to stateful variables.

Flax, Haiku and Equinox all provide APIs for initialising, getting and setting stateful values, however this requires the use of various lifting transformations, special-casing certain functions, or use of classes, and again is not easily applicable to existing code.

Our approach is most similar to the first option, where we thread state through an existing function. By using a custom interpreter acting on the function's Jaxpr, we can automate this process.

This gives rise to a nice set of features:
- minimal changes to existing code,
- full, automatic compatibility with other JAX transformations<sup>[1](#footnote1)</sup>, including third party libraries,
- no special-casing, classes, init/apply etc.

`save_inter`, `get_state` and `set_state` are all implemented as custom primitives, which by default act as no-ops

---

<a name="footnote1">1. Note, this still relies on higher-order primitives being handled in Statax.</a>

### Gotchas

- As the interpreters first trace their inputs to Jaxprs, the resulting transformed function will not include any of the original function's python side effects. This is the case in other transformations like `jit`.
- `get_state` and `set_state` only take effect once they are wrapped in a `stateful` transformation. This means that calling the untransformed function will not propagate state. `set_state` will have no effect, and all `get_state`s will rely on their initialisers. No state dictionary will be output/ accepted as an input argument.
- Caching of transformed functions, specialised to the shapes and types of the arguments, is provided as an option in config (`statax.default_config.caching`, default `True`). This is particularly important in the case where we transform other cached functions, i.e. `stateful` applied to a JIT'ed function without caching will cause the function to be recompiled every time it is called. This caching currently errs on the side of caution. The interpreter may retrace the function if the same set of arguments is passed by position vs as keywords. 
- It it possible to change the type of a stateful variable with `set_state`. `stateful` will ensure that the shape of a state variable accessed by `get_state` matches the expected shape given by its initialiser. However, taking an output state dictionary from one call to a function and feeding it back into a second call to the same function could break if the type of any state changes in the first call.
- Initialisers in `get_state` should be pure, deterministic functions. These functions are run once at trace time, and potentially multiple times thereafter to initialise state input. They may depend only on shape/ dtypes of local variables, although breakages are possible if you rely on these initialisers at runtime - ideally they only work to trace the function.

### Examples

More examples are available in the [examples](examples) folder.