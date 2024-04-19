import torch
import numpy as np
import time
import triton
import triton.language as tl
from triton.runtime.jit import TensorWrapper, reinterpret
from jax_func import associative_scan

int_dtypes = ["int8", "int16", "int32", "int64"]
uint_dtypes = ["uint8", "uint16", "uint32", "uint64"]
float_dtypes = ["float16", "float32", "float64"]
dtypes = int_dtypes + uint_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ["bfloat16"]
torch_dtypes = ["bool"] + int_dtypes + ["uint8"] + float_dtypes + ["bfloat16"]


def to_triton(x: np.ndarray, device="cuda", dst_type=None):
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip("u")  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(
            torch.tensor(x_signed, device=device).contiguous(), getattr(tl, t)
        )
    else:
        if dst_type and "float8" in dst_type:
            return reinterpret(
                torch.tensor(x, device=device).contiguous(), getattr(tl, dst_type)
            )
        if t == "float32" and dst_type == "bfloat16":
            return torch.tensor(x, device=device).contiguous().bfloat16()
        return torch.tensor(x, device=device).contiguous()


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        # FIXME: torch_dtype_name doesn't exist
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


if __name__ == "__main__":
    use_gpu = True

    if use_gpu:
        device = torch.device("cuda:0")
    else:
        device = None

    triton_times = []
    loop_times = []
    loop_comp_times = []
    jax_compat_times = []

    print("Initializing")
    op = "cumsum"
    num_warps = 16

    dim = 1
    seq_len = 2048
    batch = 4

    dtype_str = "float32"
    axis = 0
    shape = (batch, seq_len, dim)
    n_timings = 10000

    x = np.random.rand(*shape).astype(dtype=np.float32)
    inp = torch.tensor(x, device=device, requires_grad=True, dtype=torch.float32)
    init = torch.zeros(shape[1], 1, device=device, requires_grad=True)
    inp_scan = inp

    @triton.jit
    def sum_op(a, b):
        return a + b

    @triton.jit
    def kernel(X, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        x = tl.load(X + range_m[:, None] * BLOCK_N + range_n[None, :])
        # tl.device_print("z", x)
        z = tl.associative_scan(x, 0, sum_op)
        # tl.device_print("z", z)
        tl.store(Z + range_m[:, None] * BLOCK_N + range_n[None, :], z)

    print("Triton")
    z = np.empty_like(x)
    x_tri = to_triton(x, device=device)
    numpy_op = np.cumsum
    z_dtype_str = dtype_str
    z_ref = numpy_op(x, axis=axis).astype(getattr(np, z_dtype_str))
    # triton result
    z_tri = to_triton(z, device=device)
    val = kernel[(1,)](
        x_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], AXIS=axis, num_warps=num_warps
    )
    out_triton = to_numpy(z_tri)

    for _ in range(n_timings):
        # print('.', end='', flush=True)
        start = time.monotonic_ns()
        kernel[(1,)](
            x_tri,
            z_tri,
            BLOCK_M=shape[0],
            BLOCK_N=shape[1],
            AXIS=axis,
            num_warps=num_warps,
        )
        stop = time.monotonic_ns()
        triton_times.append((stop - start) / (10**9))

    print("\nFake scan")

    def f(carry, x):
        return carry + x, carry + x

    def _fake_scan(f, init, x):
        zs = []
        carry = init
        for xp in x:
            carry, out = f(carry, xp)
            zs.append(out)
        return carry, torch.stack(zs)

    expected_carry_out, expected_ys = _fake_scan(f, init, inp_scan)

    for _ in range(n_timings):
        # print('.', end='', flush=True)
        start = time.monotonic_ns()
        expected_carry_out, expected_ys = _fake_scan(f, init, inp_scan)
        stop = time.monotonic_ns()
        loop_times.append((stop - start) / (10**9))

    # _fake_scan_comp = torch.compile(_fake_scan, mode='reduce-overhead', fullgraph=True, dynamic=False)

    # # Warm-up cycles
    # print("\nFake scan-compiled")
    # for _ in range(5):
    #     expected_carry_out_comp, expected_ys_comp = _fake_scan_comp(f, init, inp_scan)

    # for _ in range(n_timings):
    #     print('.', end='', flush=True)
    #     start = time.monotonic_ns()
    #     expected_carry_out_comp, expected_ys_comp = _fake_scan_comp(f, init, inp_scan)
    #     stop = time.monotonic_ns()
    #     loop_comp_times.append((stop - start) / (10 ** 9))

    def sum_op2(a, b):
        return a + b, a + b

    # Warm-up
    print("\njax_compat")
    for _ in range(5):
        expected_ys_comp = associative_scan(sum_op2, inp_scan, axis=-1)

    for _ in range(n_timings):
        # print('.', end='', flush=True)
        start = time.monotonic_ns()
        expected_ys_comp = associative_scan(sum_op2, inp_scan, axis=-1)
        stop = time.monotonic_ns()
        jax_compat_times.append((stop - start) / (10**9))

    print()
    print("Times regular loop " + str(np.array(loop_times).mean()))
    # print('Times compiled loop ' + str(np.array(loop_comp_times).mean()))
    print("Times triton " + str(np.array(triton_times).mean()))
    print("Times jax_compat " + str(np.array(jax_compat_times).mean()))
    print("Script ended")
