import torch
import numpy as np
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing import (
    overload,
    Callable,
    Iterable,
    List,
    TypeVar,
    Any,
    Literal,
    Sequence,
    Optional,
)
from functools import partial
import math

"""
Jax-Pytorch ported functions, mostly interfaces are kept the same but unsupported features are removed:
* Jax-Keyed RNGs are sampled from global RNG
* Canonical/Named shapes/dtypes/etc are now regular shapes,dtypes
"""

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


@overload
def safe_map(f: Callable[[T1], T], __arg1: Iterable[T1]) -> List[T]: ...


@overload
def safe_map(
    f: Callable[[T1, T2], T], __arg1: Iterable[T1], __arg2: Iterable[T2]
) -> List[T]: ...


@overload
def safe_map(
    f: Callable[[T1, T2, T3], T],
    __arg1: Iterable[T1],
    __arg2: Iterable[T2],
    __arg3: Iterable[T3],
) -> List[T]: ...


@overload
def safe_map(
    f: Callable[..., T],
    __arg1: Iterable[Any],
    __arg2: Iterable[Any],
    __arg3: Iterable[Any],
    __arg4: Iterable[Any],
    *args,
) -> List[T]: ...


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(map(f, *args))


def combine(tree, operator, a_flat, b_flat):
    # Lower `fn` to operate on flattened sequences of elems.
    a = tree_unflatten(a_flat, tree)
    b = tree_unflatten(b_flat, tree)
    c = operator(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat


def _scan(tree, operator, elems, axis: int):
    """Perform scan on `elems`."""
    num_elems = elems[0].shape[axis]

    if num_elems < 2:
        return elems

    # Combine adjacent pairs of elements.
    reduced_elems = combine(
        tree,
        operator,
        [torch.ops.aten.slice(elem, axis, 0, -1, 2) for elem in elems],
        [torch.ops.aten.slice(elem, axis, 1, None, 2) for elem in elems],
    )

    # Recursively compute scan for partially reduced tensors.
    odd_elems = _scan(tree, operator, reduced_elems, axis)

    if num_elems % 2 == 0:
        even_elems = combine(
            tree,
            operator,
            [torch.ops.aten.slice(e, axis, 0, -1) for e in odd_elems],
            [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems],
        )
    else:
        even_elems = combine(
            tree,
            operator,
            odd_elems,
            [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems],
        )

    # The first element of a scan is the same as the first element
    # of the original `elems`.
    even_elems = [
        (
            torch.cat([torch.ops.aten.slice(elem, axis, 0, 1), result], dim=axis)
            if result.shape.numel() > 0 and elem.shape[axis] > 0
            else (
                result
                if result.shape.numel() > 0
                else torch.ops.aten.slice(elem, axis, 0, 1)
            )
        )  # Jax allows/ignores concat with 0-dim, Pytorch does not
        for (elem, result) in zip(elems, even_elems)
    ]

    return list(safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))


# Pytorch impl. of jax.lax.associative_scan
def associative_scan(operator: Callable, elems, axis: int = 0, reverse: bool = False):
    # if not callable(operator):
    #     raise TypeError("lax.associative_scan: fn argument should be callable.")
    elems_flat, tree = tree_flatten(elems)

    if reverse:
        elems_flat = [torch.flip(elem, [axis]) for elem in elems_flat]

    assert (
        axis >= 0 or axis < elems_flat[0].ndim
    ), "Axis should be within bounds of input"
    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same "
            "first dimension. (saw: {})".format([elem.shape for elem in elems_flat])
        )

    scans = _scan(tree, operator, elems_flat, axis)

    if reverse:
        scans = [torch.flip(scanned, [axis]) for scanned in scans]

    return tree_unflatten(scans, tree)


def test_associative_scan(shape=(1, 24, 24)):
    import jax.lax
    import jax

    x = np.random.randn(*shape)
    jx = jax.numpy.array(x)
    tx = torch.tensor(x, dtype=torch.float32)

    def nested_func(a, b):
        a_i, b_i = a
        a_j, b_j = b
        return a_j * a_i, a_j * b_i + b_j

    jy1, jy2 = jax.lax.associative_scan(nested_func, (jx, jx))
    ty1, ty2 = associative_scan(nested_func, (tx, tx))
    assert (
        np.isclose(ty1.numpy(), np.array(jy1)).all()
        and np.isclose(ty2.numpy(), np.array(jy2)).all()
    ), "Expected jax & pytorch impl to be close"

    jy1, jy2 = jax.lax.associative_scan(nested_func, (jx, jx), reverse=True)
    ty1, ty2 = associative_scan(nested_func, (tx, tx), reverse=True)
    assert (
        np.isclose(ty1.numpy(), np.array(jy1)).all()
        and np.isclose(ty2.numpy(), np.array(jy2)).all()
    ), "Expected jax & pytorch reverse impl to be close"

    print("Associative scan working as expected!")


def _interleave(a, b, axis: int):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    b_trunc = a.shape[axis] == b.shape[axis] + 1
    if b_trunc:
        pad = [0, 0] * b.ndim
        pad[(b.ndim - axis - 1) * 2 + 1] = (
            1  # +1=always end of dim, pad-order is reversed so start is at end
        )
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=axis + 1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis + 1)
    if b_trunc:
        # TODO: find torch alternative for slice_along axis for torch.jit.script to work
        interleaved = torch.ops.aten.slice(
            interleaved, axis, 0, b.shape[axis] + a.shape[axis] - 1
        )
    return interleaved


def test_interleave():
    x, y = torch.randn(1, 32, 32), torch.randn(1, 32, 32)
    v = _interleave(x, y, axis=1)
    assert v.shape == (1, 64, 32)
    assert (v[:, 0] == x[:, 0]).all()
    assert (v[:, 1] == y[:, 0]).all()
    assert (v[:, 2] == x[:, 1]).all()
    assert (v[:, 3] == y[:, 1]).all()
    assert (v[:, 4] == x[:, 2]).all()

    v = _interleave(x, y, axis=2)
    assert v.shape == (1, 32, 64)
    assert (v[..., 0] == x[..., 0]).all()
    assert (v[..., 1] == y[..., 0]).all()
    assert (v[..., 2] == x[..., 1]).all()
    assert (v[..., 3] == y[..., 1]).all()
    assert (v[..., 4] == x[..., 2]).all()

    x, y = torch.randn(1, 24, 24), torch.randn(1, 24, 24)
    assert _interleave(x, y, axis=1).shape == (1, 48, 24)
    assert _interleave(x, y, axis=2).shape == (1, 24, 48)

    x, y = torch.randn(3, 96), torch.randn(2, 96)
    v = _interleave(x, y, axis=0)
    assert v.shape == (5, 96)
    assert (v[0] == x[0]).all()
    assert (v[1] == y[0]).all()
    assert (v[2] == x[1]).all()
    assert (v[3] == y[1]).all()
    assert (v[4] == x[2]).all()
    print("Interleave working as expected!")


def _compute_fans(shape, fan_in_axes=None):
    """Computes the number of input and output units for a weight shape."""
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        if fan_in_axes is not None:
            # Compute fan-in using user-specified fan-in axes.
            fan_in = np.prod([shape[i] for i in fan_in_axes])
            fan_out = np.prod([s for i, s in enumerate(shape) if i not in fan_in_axes])
        else:
            # If no axes specified, assume convolution kernels (2D, 3D, or more.)
            # kernel_shape: (..., input_depth, depth)
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


def uniform(shape, dtype=torch.float, minval=0.0, maxval=1.0, device=None):
    src = torch.rand(shape, dtype=dtype, device=device)
    if minval == 0 and maxval == 1.0:
        return src
    else:
        return (src * (maxval - minval)) + minval


def _complex_uniform(shape: Sequence[int], dtype, device=None) -> torch.Tensor:
    """
    Sample uniform random values within a disk on the complex plane,
    with zero mean and unit variance.
    """
    r = torch.sqrt(2 * torch.rand(shape, dtype=dtype, device=device))
    theta = 2 * torch.pi * torch.rand(shape, dtype=dtype, device=device)
    return r * torch.exp(1j * theta)


def complex_as_float_dtype(dtype):
    match dtype:
        case torch.complex32:
            return torch.float32  # NOTE: complexe32 is not wel supported yet
        case torch.complex64:
            return torch.float32
        case torch.complex128:
            return torch.float64
        case _:
            return dtype


def _complex_truncated_normal(
    upper: float, shape: Sequence[int], dtype, device=None
) -> torch.Tensor:
    """
    Sample random values from a centered normal distribution on the complex plane,
    whose modulus is truncated to `upper`, and the variance before the truncation
    is one.
    """
    real_dtype = torch.tensor(0, dtype=dtype).real.dtype
    t = (
        1 - torch.exp(torch.tensor(-(upper**2), dtype=dtype, device=device))
    ) * torch.rand(shape, dtype=real_dtype, device=device).type(dtype)
    r = torch.sqrt(-torch.log(1 - t))
    theta = (
        2 * torch.pi * torch.rand(shape, dtype=real_dtype, device=device).type(dtype)
    )
    return r * torch.exp(1j * theta)


def _truncated_normal(lower, upper, shape, dtype=torch.float):
    if shape is None:
        shape = torch.broadcast_shapes(np.shape(lower), np.shape(upper))

    sqrt2 = math.sqrt(2)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)

    # a<u<b
    u = uniform(shape, dtype, minval=a, maxval=b)
    out = sqrt2 * torch.erfinv(u)
    # Clamp the value to the open interval (lower, upper) to make sure that
    # rounding (or if we chose `a` for `u`) doesn't push us outside of the range.
    with torch.no_grad():
        return torch.clip(
            out,
            torch.nextafter(torch.tensor(lower), torch.tensor(np.inf, dtype=dtype)),
            torch.nextafter(torch.tensor(upper), torch.tensor(-np.inf, dtype=dtype)),
        )


def variance_scaling(
    scale: float,
    mode: Literal["fan_in", "fan_out", "fan_avg"] = "fan_in",
    distribution: Literal["truncated_normal", "normal", "uniform"] = "truncated_normal",
    fan_in_axes: Optional[Sequence[int]] = None,
    dtype=torch.float,
):
    def init(shape: Sequence[float], dtype=dtype, device=None):
        fan_in, fan_out = _compute_fans(shape, fan_in_axes)
        match mode:
            case "fan_in":
                denom = max(1, fan_in)
            case "fan_out":
                denom = max(1, fan_out)
            case "fan_avg":
                denom = max(1, (fan_in + fan_out) / 2)
            case _:
                raise ValueError(
                    f"invalid mode for variance scaling initializer: {mode}"
                )

        variance = scale / denom
        match distribution:
            case "normal":
                return torch.normal(
                    0, np.sqrt(variance), shape, dtype=dtype, device=device
                )
            case "uniform":
                if dtype.is_complex:
                    return _complex_uniform(
                        shape, dtype=dtype, device=device
                    ) * np.sqrt(variance)
                else:
                    return uniform(
                        shape, dtype=dtype, device=device, minval=-1, maxval=1.0
                    ) * np.sqrt(3 * variance)
            case "truncated_normal":
                if dtype.is_complex:
                    stddev = np.sqrt(variance) * 0.95311164380491208
                    return (
                        _complex_truncated_normal(2, shape, dtype=dtype, device=device)
                        * stddev
                    )
                else:
                    stddev = np.sqrt(variance) * 0.87962566103423978
                    return _truncated_normal(-2.0, 2.0, shape, dtype=dtype) * stddev
            case _:
                raise ValueError(
                    f"invalid distribution for variance scaling initializer: {distribution}"
                )

    return init


def lecun_normal(fan_in_axes=None, dtype=torch.float):
    """Builds a Lecun normal initializer.

    A `Lecun normal initializer`_ is a specialization of
    :func:`jax.nn.initializers.variance_scaling` where ``scale = 1.0``,
    ``mode="fan_in"``, and ``distribution="truncated_normal"``.

    Args:
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.

    Returns:
    An initializer.

    Example:

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.lecun_normal()
    >>> initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)  # doctest: +SKIP
    Array([[ 0.46700746,  0.8414632 ,  0.8518669 ],
         [-0.61677957, -0.67402434,  0.09683388]], dtype=float32)

    .. _Lecun normal initializer: https://arxiv.org/abs/1706.02515
    """
    return variance_scaling(
        1.0, "fan_in", "truncated_normal", fan_in_axes=fan_in_axes, dtype=dtype
    )


def test_variance_scaling():
    v = variance_scaling(1.0, distribution="normal")
    n_f32 = v((1, 10000), dtype=torch.float)
    assert np.isclose(
        n_f32.std().item(), 1.0, rtol=0.015, atol=0.015
    ), f"std for f32 normal[0,1.0] is {n_f32.std()} != 1.0"
    del n_f32
    # NOTE: this is used in the original as `complex_normal` (but with stddev=0.5**0.5)
    n_c64 = v((1, 10000), dtype=torch.complex64)
    assert np.isclose(
        n_c64.std().item(), 1.0, rtol=0.015, atol=0.015
    ), f"std for c64 normal[0,1.0] is {n_c64.std()} != 1.0"
    del n_c64

    # Truncated normal
    v = variance_scaling(1.0, distribution="truncated_normal")
    tn_f32 = v((1, 10000), dtype=torch.float)
    assert np.isclose(
        tn_f32.std().item(), 0.775, rtol=0.015, atol=0.015
    ), f"std for f32 truncated normal[0,1.0] is {tn_f32.std()} != 0.775"
    del tn_f32

    # NOTE: this is used in the original (both trunc_standard_normal & lecun_normal it seems),
    # seems that they are using the fan-in/out feature to 'hide the low variance initialization'
    # The actual std observed is np.sqrt(2/shape[1]/(2*shape[0])); shape[2] has no impact
    v = variance_scaling(1.0, distribution="truncated_normal")
    tn_f32 = v((1, 10000, 2), dtype=torch.float)
    tn_c32 = torch.complex(tn_f32[..., 0], tn_f32[..., 1])
    expected_std = np.sqrt(2 / tn_f32.shape[1] / (2 * tn_f32.shape[0]))
    print(tn_c32.shape)
    assert np.isclose(
        tn_c32.std().item(), expected_std, rtol=0.015, atol=0.015
    ), f"std for f32 truncated normal[0,1.0] is {tn_c32.std()} != {expected_std}"
    del tn_f32
    del tn_c32

    print("Variance scaling working as expected!")


if __name__ == "__main__":
    test_variance_scaling()
    test_interleave()
    test_associative_scan()
    test_associative_scan(shape=(2, 256, 24))
    test_associative_scan(shape=(360, 96))
