import torch
import numpy as np
from .jax_func import variance_scaling, lecun_normal, uniform
import scipy.linalg

# Initialization Functions


def make_HiPPO(N):
    """Create a HiPPO-LegS matrix.
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation
    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = np.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def make_Normal_S(N):
    nhippo = make_HiPPO(N)
    # Add in a rank 1 term. Makes it Normal.
    p = 0.5 * np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, np.newaxis] * q[np.newaxis, :]
    return S


def make_Normal_HiPPO(N, B=1):
    """Create a normal approximation to HiPPO-LegS matrix.
    For HiPPO matrix A, A=S+pqT is normal plus low-rank for
    a certain normal matrix S and low rank terms p and q.
    We are going to approximate the HiPPO matrix with the normal matrix S.
    Note we use original numpy instead of jax.numpy first to use the
    onp.linalg.eig function. This is because Jax's linalg.eig function does not run
    on GPU for non-symmetric matrices. This creates tracing issues.
    So we instead use onp.linalg eig and then cast to a jax array
    (since we only have to do this once in the beginning to initialize).
    Args:
        N (int32): state size
        B (int32): diagonal blocks
    Returns:
        Lambda (complex64): eigenvalues of S (N,)
        V      (complex64): eigenvectors of S (N,N)
    """

    assert N % B == 0, "N must divide blocks"
    S = (make_Normal_S(N // B),) * B
    S = scipy.linalg.block_diag(*S)

    # Diagonalize S to V \Lambda V^*
    Lambda, V = np.linalg.eig(S)

    # Convert to jax array
    return torch.tensor(Lambda), torch.tensor(V)


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """Initialize the learnable timescale Delta by sampling
    uniformly between dt_min and dt_max.
    Args:
        dt_min (float32): minimum value
        dt_max (float32): maximum value
    Returns:
        init function
    """

    def init(shape):
        """Init function
        Args:
            key: jax random key
            shape tuple: desired shape
        Returns:
            sampled log_step (float32)
        """
        return uniform(shape, minval=np.log(dt_min), maxval=np.log(dt_max))
        # return torch.rand(shape) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)

    return init


def init_log_steps(H, dt_min, dt_max):
    """Initialize an array of learnable timescale parameters
    Args:
        key: jax random key
        input: tuple containing the array shape H and
               dt_min and dt_max
    Returns:
        initialized array of timescales (float32): (H,)
    """
    log_steps = []
    for i in range(H):
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(shape=(1,))
        log_steps.append(log_step)

    return torch.tensor(log_steps)


def init_VinvB(init_fun, Vinv):
    """Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         init_fun:  the initialization function to use, e.g. lecun_normal()
         shape (tuple): desired shape  (P,H)
         Vinv: (complex64)     the inverse eigenvectors used for initialization
     Returns:
         B_tilde (complex64) of shape (P,H,2)
    """

    def init(shape, dtype):
        B = init_fun(shape, dtype)
        VinvB = Vinv @ B.type(Vinv.dtype)
        VinvB_real = VinvB.real
        VinvB_imag = VinvB.imag
        return torch.cat((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)

    return init


def trunc_standard_normal(shape):
    """Sample C with a truncated normal distribution with standard deviation 1.
    Args:
        key: jax random key
        shape (tuple): desired shape, of length 3, (H,P,_)
    Returns:
        sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
    """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        C = lecun_normal()(shape=(1, P, 2))
        Cs.append(C)
    return torch.tensor(Cs)[:, 0]


def init_CV(init_fun, shape, V) -> torch.Tensor:
    """Initialize C_tilde=CV. First sample C. Then compute CV.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         init_fun:  the initialization function to use, e.g. lecun_normal()
         shape (tuple): desired shape  (H,P)
         V: (complex64)     the eigenvectors used for initialization
     Returns:
         C_tilde (complex64) of shape (H,P,2)
    """
    C_ = init_fun(shape + (2,))
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    return CV


def init_columnwise_B(shape, dtype):
    """Initialize B matrix in columnwise fashion.
    We will sample each column of B from a lecun_normal distribution.
    This gives a different fan-in size then if we sample the entire
    matrix B at once. We found this approach to be helpful for PathX
    It appears to be related to the point in
    https://arxiv.org/abs/2206.12037 regarding the initialization of
    the C matrix in S4, so potentially more important for the
    C initialization than for B.
     Args:
         key: jax random key
         shape (tuple): desired shape, either of length 3, (P,H,_), or
                      of length 2 (N,H) depending on if the function is called
                      from the low-rank factorization initialization or a dense
                      initialization
     Returns:
         sampled B matrix (float32), either of shape (H,P) or
          shape (H,P,2) (for complex parameterization)
    """
    shape = shape[:2] + ((2,) if len(shape) == 3 else ())
    lecun = variance_scaling(0.5 if len(shape) == 3 else 1.0, fan_in_axes=(0,))
    return lecun(shape, dtype)


def init_columnwise_VinvB(init_fun, Vinv):
    """Same function as above, but with transpose applied to prevent shape mismatch
    when using the columnwise initialization. In general this is unnecessary
    and will be removed in future versions, but is left for now consistency with
    certain random seeds until we rerun experiments."""

    def init(shape, dtype):
        B = init_fun(shape[:2], dtype)
        VinvB = Vinv @ B
        VinvB_real = VinvB.real
        VinvB_imag = VinvB.imag
        return torch.cat((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)

    return init


def init_rowwise_C(shape, dtype):
    """Initialize C matrix in rowwise fashion. Analogous to init_columnwise_B function above.
    We will sample each row of C from a lecun_normal distribution.
    This gives a different fan-in size then if we sample the entire
    matrix B at once. We found this approach to be helpful for PathX.
    It appears to be related to the point in
    https://arxiv.org/abs/2206.12037 regarding the initialization of
    the C matrix in S4.
     Args:
         shape (tuple): desired shape, of length 3, (H,P,_)
     Returns:
         sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
    """
    shape = shape[:2] + ((2,) if len(shape) == 3 else ())
    lecun = variance_scaling(0.5, fan_in_axes=(0,))
    return lecun(shape, dtype)
