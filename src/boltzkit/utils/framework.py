from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    TypeVar,
    Union,
)
import numpy as np
import warnings

if TYPE_CHECKING:
    # Import modules for type-hinting (okay if not installed)
    import torch
    import jax

    Array = Union[np.ndarray, "torch.Tensor", "jax.Array"]

GenericArrayType = TypeVar("T", np.ndarray, "torch.Tensor", "jax.Array")


def is_torch_tensor(x) -> bool:
    try:
        global torch
        import torch
    except ImportError:
        return False
    return isinstance(x, torch.Tensor)


def is_jax_array(x) -> bool:
    try:
        global jax
        import jax
    except ImportError:
        return False
    return isinstance(x, jax.Array)


FrameworkName = Literal["numpy", "jax", "pytorch"]


def detect_framework(x: "Array") -> FrameworkName:
    if isinstance(x, np.ndarray):
        return "numpy"
    elif is_torch_tensor(x):
        return "pytorch"
    elif is_jax_array(x):
        return "jax"
    else:
        raise ValueError(
            f"Framework could not be detected or imported for input of type '{type(x).__name__}'."
        )


def create_pytorch_value_and_grad_fn(
    value_fn: Callable[["torch.Tensor"], "torch.Tensor"],
):
    """Computes per-sample gradients of the first output of value_fn with respect to its first input."""
    import torch

    def value_and_grad_fn(x: torch.Tensor):
        x_leaf = x.detach().requires_grad_(True)
        value_leaf = value_fn(x_leaf)
        if isinstance(value_leaf, (list, tuple)):
            first_value = value_leaf[0]
            assert isinstance(first_value, torch.Tensor)
        else:
            first_value = value_leaf

        grad_outputs = torch.ones_like(first_value)
        grad = torch.autograd.grad(
            first_value, x_leaf, grad_outputs=grad_outputs, create_graph=True
        )[0]

        value = value_fn(x)
        return value, grad.detach()

    return value_and_grad_fn


def to_numpy(x: "Array", source_framework: FrameworkName) -> np.ndarray:
    if source_framework == "numpy":
        return x
    elif source_framework == "pytorch":
        return x.detach().cpu().numpy()
    elif source_framework == "jax":
        global jax
        import jax

        return jax.device_get(x)
    else:
        raise ValueError(f"Unknown framework of name '{source_framework}'")


def to_numpy_recursive(x: Any, source_framework: FrameworkName):
    if isinstance(x, list):
        return [to_numpy_recursive(e, source_framework) for e in x]
    elif isinstance(x, tuple):
        return tuple([to_numpy_recursive(e, source_framework) for e in x])
    elif isinstance(x, dict):
        return {k: to_numpy_recursive(e, source_framework) for k, e in x.items()}
    else:
        return to_numpy(x, source_framework)


def from_numpy(
    x: np.ndarray,
    target_framework: FrameworkName,
    style=None,
):
    if style is not None:
        device, dtype = style
    else:
        device, dtype = None, None
    if target_framework == "numpy":
        if dtype is None:
            np_dtype = None
        else:
            np_dtype = dtype
        return np.asarray(x, dtype=np_dtype)
    elif target_framework == "pytorch":
        return torch.tensor(x, dtype=dtype, device=device)
    elif target_framework == "jax":
        global jnp
        import jax.numpy as jnp

        return jnp.asarray(x)
    else:
        raise ValueError(f"Unknown framework of name '{target_framework}'")


def from_numpy_recursive(x: Any, target_framework: FrameworkName, style=None):
    if isinstance(x, np.ndarray):
        return from_numpy(x, target_framework, style=style)
    if isinstance(x, list):
        return [from_numpy_recursive(e, target_framework, style=style) for e in x]
    elif isinstance(x, tuple):
        return tuple(
            [from_numpy_recursive(e, target_framework, style=style) for e in x]
        )
    elif isinstance(x, dict):
        return {
            k: from_numpy_recursive(e, target_framework, style=style)
            for k, e in x.items()
        }
    else:
        raise ValueError(f"Unsupported input type '{type(x).__name__}'")


def _create_pytorch_autograd_func(
    value_and_grad_fn: Callable[
        ["torch.Tensor"], tuple["torch.Tensor", "torch.Tensor"]
    ],
    include_grad: bool = False,
):
    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            value, grad = value_and_grad_fn(x)
            ctx.save_for_backward(grad)
            # Return log_probs as torch tensor

            if include_grad:
                return value, grad.detach()
            else:
                return value

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor, *args):
            (grad,) = ctx.saved_tensors
            # grad_output has shape (batch,) and scores has shape (batch, dim)
            # We want dL/dx = dL/dlogp * dlogp/dx
            grad_input = grad_output.unsqueeze(1) * grad
            return grad_input

    return Function


class FrameworkAgnosticFunction:
    def __init__(
        self,
        impl_framework: FrameworkName,
        value_fn: Callable[["Array"], "Array"],
        grad_fn: None | Callable[["Array"], "Array"] = None,
        value_and_grad_fn: None | Callable[["Array"], tuple["Array", "Array"]] = None,
    ):
        assert (grad_fn is None) == (value_and_grad_fn is None)

        self._impl_framework: FrameworkName = impl_framework
        self._value_fn = value_fn
        self._grad_fn = grad_fn
        self._value_and_grad_fn = value_and_grad_fn

    def __call__(self, x: "Array"):
        return self.get_value(x)

    def get_value(self, x: "Array") -> "Array":
        if is_torch_tensor(x) and x.requires_grad and self._impl_framework != "pytorch":
            f = _create_pytorch_autograd_func(self._value_and_grad_fn)
            return f.apply(x)
        else:
            return self._value_fn(x)

    def get_grad(self, x: "Array") -> "Array":
        if self._grad_fn is None:
            raise NotImplementedError("Gradients are not supported for this function")
        grad = self._grad_fn(x)
        if is_torch_tensor(grad):
            # autograd through gradients are not supported
            grad = grad.detach()
        return grad

    def get_value_and_grad(self, x: "Array") -> tuple["Array", "Array"]:
        if self._value_and_grad_fn is None:
            raise NotImplementedError("Gradients are not supported for this function")

        if is_torch_tensor(x) and x.requires_grad and self._impl_framework != "pytorch":
            f = _create_pytorch_autograd_func(
                self._value_and_grad_fn, include_grad=True
            )
            value, grad = f.apply(x)
        else:
            value, grad = self._value_and_grad_fn(x)

        if is_torch_tensor(grad):
            # autograd through gradients are not supported
            grad = grad.detach()
        return value, grad


def make_agnostic_simple(*, implementation: FrameworkName):
    def decorator(
        value_fn: Callable[["Array"], "Array"],
    ):
        # Framework agnostic value function
        def wrapped_value_fn(x: "Array"):
            detected_framework = detect_framework(x)

            if detected_framework == implementation:
                return value_fn(x)

            x_np = to_numpy(x, detected_framework)
            x_impl = from_numpy(x_np, implementation)
            out_impl = value_fn(x_impl)
            out_np = to_numpy_recursive(out_impl, implementation)

            if is_torch_tensor(x):
                device = x.device
            else:
                device = None
            dtype = x.dtype

            out = from_numpy_recursive(
                out_np, detected_framework, style=(device, dtype)
            )
            return out

        return wrapped_value_fn

    return decorator


T = TypeVar("T")


def try_jit_jax(f: T) -> T:
    global jax
    import jax

    try:
        return jax.jit(f)
    except Exception as e:
        warnings.warn(f"Failed to jit function ({repr(e)})")
        return f


def make_agnostic(
    *,
    implementation: FrameworkName,
    grad_fn: None | Callable[["Array"], "Array"] = None,
    value_and_grad_fn: None | Callable[["Array"], tuple["Array", "Array"]] = None,
    value_fn: None | Callable[["Array"], "Array"] = None,
):
    """
    Make a function agnostic to NumPy, JAX, or PyTorch.

    This function takes a `value_fn` already implemented in one of these frameworks
    and returns a uniform, framework-agnostic wrapper. Optionally, a gradient function
    (`grad_fn`) or a combined value-and-gradient function (`value_and_grad_fn`) can
    be provided, which is useful when:

    - Automatic differentiation is not available (e.g., NumPy), or
    - An analytical gradient exists, which may be more efficient than computing it automatically.

    The resulting wrapper can be used either as a **decorator** (requires `value_fn` to be `None`):


    ```python
    import numpy as np

    def log_prob_grad(x: np.ndarray):
        return -2 * x

    @make_agnostic(implementation="numpy", grad_fn=log_prob_grad)
    def log_prob(x: np.ndarray):
        return -np.sum(x**2, axis=-1)
    ```

    or as a **factory/wrapper function** (requires `value_fn` to be specified):

    ```python
    import numpy as np

    def log_prob_grad(x: np.ndarray):
        return -2 * x

    def log_prob_val(x: np.ndarray):
        return -np.sum(x**2, axis=-1)

    log_prob = make_agnostic(
        implementation="numpy",
        value_fn=log_prob_val,
        grad_fn=log_prob_grad
    )
    ```

    Parameters
    ----------
    implementation : FrameworkName
        The framework in which the function is implemented. Supported values:
        `"numpy"`, `"jax"`, or `"pytorch"`.
    grad_fn : callable, optional
        A function that computes the gradient of `value_fn`. Signature should be:
        `grad_fn(x: Array) -> Array`.
    value_and_grad_fn : callable, optional
        A function returning both the value and gradient. Signature should be:
        `value_and_grad_fn(x: Array) -> Tuple[Array, Array]`.
        Providing this may be more efficient than computing value and gradient separately.
    value_fn : callable, optional
        The primary function to wrap. Signature should be: `value_fn(x: Array) -> Array`.
        - If `value_fn` is provided directly as an argument, `make_agnostic` acts as a **factory/wrapper** and returns the wrapped function immediately.
        - If `value_fn` is `None` (default), `make_agnostic` acts as a **decorator** that can be applied to a function later.


    Returns
    -------
    FrameworkAgnosticFunction
        A framework-agnostic wrapper around the provided functions. If `value_fn` is
        provided directly, returns the wrapped function immediately; otherwise, returns
        a decorator that can be applied to a function.


    Framework-specific caveats
    --------------------------
    - **NumPy:** Automatic differentiation is not available. The user should provide
    `grad_fn`, `value_and_grad_fn`, or both if gradients are needed.
    - **JAX:** Functions are automatically JIT-compiled and vectorized when possible.
    They should therefore be provided in a **non-vectorized** form.
    - **PyTorch:** First-order automatic differentiation is supported. Gradients can
    flow through `value_fn`, even if implemented in another framework. Requirement is that the input is
    of type `torch.Tensor`. This does not translate to Jax input as `jax.grad(f)` requires f to be a non-batched function,
    which would result in many individual calls to `value_fn` when using `jax.vmap(jax.grad(f))(batch)`.
    """

    if implementation == "jax":
        global jax
        import jax
    elif implementation == "pytorch":
        global torch
        import torch

    def decorator(
        value_fn: Callable[["Array"], "Array"],
        grad_fn=grad_fn,
        value_and_grad_fn=value_and_grad_fn,
    ):
        wrap_simple = make_agnostic_simple(implementation=implementation)

        wrapped_value_fn = None
        wrapped_grad_fn = None
        wrapped_value_and_grad_fn = None

        if implementation == "jax":
            jax_batched_value_fn = jax.vmap(value_fn)
            jax_batched_value_fn = try_jit_jax(jax_batched_value_fn)
            wrapped_value_fn = wrap_simple(jax_batched_value_fn)

            if grad_fn is None:
                jax_grad_fn = jax.vmap(jax.grad(value_fn))
            else:
                jax_grad_fn = jax.vmap(grad_fn)
            jax_grad_fn = try_jit_jax(jax_grad_fn)
            wrapped_grad_fn = wrap_simple(jax_grad_fn)

            if value_and_grad_fn is None:
                jax_value_and_grad_fn = jax.vmap(jax.value_and_grad(value_fn))
            else:
                jax_value_and_grad_fn = jax.vmap(value_and_grad_fn)
            jax_value_and_grad_fn = try_jit_jax(jax_value_and_grad_fn)
            wrapped_value_and_grad_fn = wrap_simple(jax_value_and_grad_fn)

        elif implementation == "pytorch":
            if grad_fn is None and value_and_grad_fn is None:
                # In this case, there is no more efficient way to compute both
                torch_value_and_grad_fn = create_pytorch_value_and_grad_fn(value_fn)
                torch_grad_fn = lambda x: torch_value_and_grad_fn(x)[1]
            elif grad_fn is None and value_and_grad_fn is not None:
                torch_value_and_grad_fn = value_and_grad_fn
                torch_grad_fn = lambda x: torch_value_and_grad_fn(x)[1]
            elif grad_fn is not None and value_and_grad_fn is None:
                torch_grad_fn = grad_fn
                torch_value_and_grad_fn = lambda x: (value_fn(x), torch_grad_fn(x))
            else:
                torch_grad_fn = grad_fn
                torch_value_and_grad_fn = value_and_grad_fn

            wrapped_grad_fn = wrap_simple(torch_grad_fn)
            wrapped_value_and_grad_fn = wrap_simple(torch_value_and_grad_fn)

        else:
            if grad_fn is not None and value_and_grad_fn is None:
                value_and_grad_fn = lambda x: (value_fn(x), grad_fn(x))
            elif grad_fn is None and value_and_grad_fn is not None:
                grad_fn = lambda x: value_and_grad_fn(x)[1]

            if grad_fn is not None:
                wrapped_grad_fn = wrap_simple(grad_fn)
            if value_and_grad_fn is not None:
                wrapped_value_and_grad_fn = wrap_simple(value_and_grad_fn)

        if wrapped_value_fn is None:  # default
            wrapped_value_fn = wrap_simple(value_fn)

        return FrameworkAgnosticFunction(
            impl_framework=implementation,
            value_fn=wrapped_value_fn,
            grad_fn=wrapped_grad_fn,
            value_and_grad_fn=wrapped_value_and_grad_fn,
        )

    if value_fn is not None:
        return decorator(value_fn)
    else:
        return decorator


def create_dispatch(
    impl_np: Callable | None = None,
    impl_torch: Callable | None = None,
    impl_jax: Callable | None = None,
    vmap_jax: bool = False,
    use_jit: bool = False,
):
    available_implementations: dict[FrameworkName, Callable] = {}
    if impl_np is not None:
        available_implementations["numpy"] = impl_np
    if impl_torch is not None:
        available_implementations["pytorch"] = impl_torch
    if impl_jax is not None:
        try:
            import jax
        except:
            raise ValueError(
                "Function was given a jax implementation `impl_jax` but jax could not be imported"
            )

        if vmap_jax:
            impl_jax = jax.vmap(impl_jax)
        if use_jit:
            impl_jax = jax.jit(impl_jax)

        available_implementations["jax"] = impl_jax

    def fn(*args, **kwargs):
        framework = detect_framework(args[0])
        if framework not in available_implementations:
            raise ValueError(
                f"The function was called with an array-type from framework '{framework}' ({type(args[0]).__name__}) but no {framework} implementation was provided"
            )
        fn = available_implementations[framework]
        return fn(*args, **kwargs)

    return fn


if __name__ == "__main__":
    import torch
    import jax

    # === Numpy-based implementation ===
    def foo_grad_np(x: np.ndarray):
        return -2 * x

    def foo_val_np(x: np.ndarray):
        return -np.sum(x**2, -1)

    foo_np = make_agnostic(
        implementation="numpy", value_fn=foo_val_np, grad_fn=foo_grad_np
    )

    def foo_grad_jax(x: jax.Array):
        return -2 * x

    def foo_val_jax(x: jax.Array):
        return -jax.numpy.sum(x**2, -1)

    foo_jax = make_agnostic(implementation="jax", value_fn=foo_val_jax)

    # === PyTorch-based implementation ===
    @make_agnostic(implementation="pytorch")
    def foo_torch(x: torch.Tensor):
        return -torch.sum(x**2, -1)

    x_torch = torch.asarray([[1, 2], [3, -0.5]], requires_grad=True)
    y = 2 * x_torch + x_torch**2

    val = foo_jax(y)
    val.sum().backward()
    print(x_torch.grad)

    grad = foo_torch.get_grad(jax.numpy.asarray(y.detach().numpy()))
    print(grad, type(grad))
