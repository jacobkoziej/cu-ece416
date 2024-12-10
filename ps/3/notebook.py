# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# SPDX-License-Identifier: GPL-3.0-or-later
#
# noteboook.py -- ps3: adaptive equalization
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# ruff: noqa: E402

# %%
import numpy as np

from einops import (
    einsum,
    parse_shape,
    repeat,
)
from numpy.linalg import eigvals
from scipy.linalg import toeplitz
from scipy.signal import lfilter

# %% tags=["parameters"]
ALPHA = 0.1
P_DB = -30

# %% [markdown]
# ## Experiment

# %%
N_0 = 3
M_0 = 5
M_MAX = 11
N_ITER = 20
K = int(10e4)
LAMBDA = 0.9
DELTA = 0.01

# %%
N_TRAIN = M_MAX - M_0 + N_ITER - 1

# %%
mu_v = 0
sigma_v = 10 ** (P_DB / 10)

# %%
SEED = 0x872C5047
rng = np.random.default_rng(SEED)

# %%
h = np.append(np.zeros(N_0 - 1), [ALPHA, 1, -ALPHA])

# %%
x = rng.integers(0, 2, N_TRAIN + M_0 + K)
x[x == 0] = -1

v = rng.normal(mu_v, sigma_v, N_TRAIN + M_0 + K)

y = lfilter(h, 1, x) + v

# %%
y_train = y[: N_ITER + M_MAX - 1]

A = toeplitz(np.flip(y_train[:M_MAX]), y_train[(M_MAX - 1) :])

del y_train

# %%
start = (M_MAX - M_0 - 1) - 1
end = start + N_ITER

d = x[start:end]

del start
del end

# %%
y_m_max = y[(N_TRAIN + 1 + M_0) :]
y_m_max = toeplitz(np.flip(y_m_max[:M_MAX]), y_m_max[(M_MAX - 1) :])

# %%
x_k = x[N_TRAIN + M_MAX - 1 : -(M_0 + 1)]


# %%
def rls(llambda, delta, D, A):
    shape = parse_shape(A, "M N")

    shape["N"] += 1

    N = shape["N"]
    M = shape["M"]

    P0 = (delta**-1) * np.identity(M)
    P0 = repeat(P0, "... -> N ...", N=N)

    K = np.zeros((N - 1, M))
    E = np.zeros(N - 1)
    W = np.zeros((N, M))

    A = A.T

    for a, P_prev, P, d, k, e, w_prev, w in zip(
        A,
        P0[:-1],
        P0[1:],
        D,
        K,
        np.nditer(E, op_flags=["readwrite"]),
        W[:-1],
        W[1:],
    ):
        s = P_prev @ a
        k[...] = ((llambda + (a.conj() @ s)) ** -1) * s
        e[...] = d - (w_prev.conj() @ a)
        w[...] = w_prev + (k * e.conj())
        P[...] = (llambda**-1) * (P_prev - einsum(k, s.conj(), "i, j -> i j"))

        assert (eigvals(P_prev) > 0).all()

    return W[1:].T, E, K.T, P0[-1]
