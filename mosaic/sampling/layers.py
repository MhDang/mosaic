"""Log-space matrix products and segment reductions the samplers are built from.

Every message the samplers pass lives in log space. Products of a log-space
operand with a linear (0/1) one shift by a max, exponentiate, multiply and take
the log back.

Importing this module sets ``torch.set_float32_matmul_precision("high")``
(TF32 matmuls).
"""

import torch

torch.set_float32_matmul_precision("high")


# ---- log-space matrix products ----


def matmul_a_logb(A, B):
    """``log(A @ exp(B))`` with ``A`` linear and ``B`` in log space.

    Stable: subtracts the per-column max of ``B`` before exponentiating.
    """
    bd = len(B.shape) - 2
    B_max = torch.amax(B, dim=bd, keepdim=True)
    B = B - B_max
    B.nan_to_num_(nan=float("-inf"))
    B.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(B_max)
    return C


def matmul_loga_b(A, B):
    """``log(exp(A) @ B)`` with ``A`` in log space and ``B`` linear.

    Stable: subtracts the per-row max of ``A`` before exponentiating.
    """
    A_max = torch.amax(A, dim=-1, keepdim=True)
    A_max_safe = A_max.masked_fill(A_max.isinf(), 0.0)
    A = A - A_max_safe
    A.nan_to_num_(nan=float("-inf"))
    A.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(A_max_safe)
    return C


def matmul_loga_logb(A, B, compute_dtype=None):
    """``log(exp(A) @ exp(B))`` with both operands in log space.

    Shifts ``A`` by its per-row max and ``B`` by its per-column max, multiplies
    in linear space, and adds the shifts back. The shifts are separate, so a
    term far below both maxima (about 87 nats in fp32, 745 in fp64) underflows
    to zero; ``compute_dtype=torch.float64`` does the whole product in fp64 and
    returns the input dtype.
    """
    dtype = A.dtype
    if compute_dtype is not None:
        A = A.to(compute_dtype)
        B = B.to(compute_dtype)
    A_max = A.amax(dim=-1, keepdim=True)
    A_max_safe = A_max.masked_fill(A_max.isinf(), 0.0)
    A_lin = (A - A_max_safe).exp()
    B_max = B.amax(dim=-2, keepdim=True)
    B_max_safe = B_max.masked_fill(B_max.isinf(), 0.0)
    B_lin = (B - B_max_safe).exp()
    C_lin = torch.matmul(A_lin, B_lin)
    return (C_lin.log() + A_max_safe + B_max_safe).to(dtype)


# ---- segment reductions over the last dimension ----


def scatter_logsumexp(src, index, dim_size):
    """Segment logsumexp of ``src`` (``(..., E)``, log space) into ``dim_size`` bins.

    ``index`` (``(E,)``) assigns each element to a bin; empty bins get -inf.
    """
    *prefix, E = src.shape
    device, dtype = src.device, src.dtype
    neg_inf = float("-inf")

    expanded_index = index.view(*([1] * len(prefix)), E).expand(*prefix, E)

    out_max = torch.full((*prefix, dim_size), neg_inf, device=device, dtype=dtype)
    out_max.scatter_reduce_(
        dim=-1, index=expanded_index, src=src, reduce="amax", include_self=True
    )

    gathered_max = out_max.gather(dim=-1, index=expanded_index)
    shifted = torch.exp(src - gathered_max)

    out_sum = torch.zeros((*prefix, dim_size), device=device, dtype=dtype)
    out_sum.scatter_add_(dim=-1, index=expanded_index, src=shifted)

    out = out_max + torch.log(out_sum)
    return torch.where(out_sum > 0, out, torch.full_like(out, neg_inf))
