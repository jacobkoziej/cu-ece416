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
from numpy.linalg import (
    eigvals,
    norm,
    qr,
)
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


# %%
def invqrdrls(llambda, delta, D, U):
    shape = parse_shape(U, "M N")

    shape["N"] += 1

    N = shape["N"]
    M = shape["M"]

    Pch0 = (delta ** (-1 / 2)) * np.identity(M)
    Pch0 = repeat(Pch0, "... -> N ...", N=N)

    K = np.zeros((N - 1, M))
    E = np.zeros(N - 1)
    W = np.zeros((N, M))

    gamma_inv_sqrt0 = np.zeros(N - 1)

    U = U.T

    lambda_sqrt = llambda ** (-1 / 2)

    for u, Pch_prev, Pch, d, k, e, w_prev, w, gamma_inv_sqrt in zip(
        U,
        Pch0[:-1],
        Pch0[1:],
        D,
        K,
        np.nditer(E, op_flags=["readwrite"]),
        W[:-1],
        W[1:],
        np.nditer(gamma_inv_sqrt0, op_flags=["readwrite"]),
    ):
        A = np.block(
            [
                [1, lambda_sqrt * einsum(u.conj(), Pch_prev, "i, i j -> j")],
                [np.zeros((M, 1)), lambda_sqrt * Pch_prev],
            ]
        )

        _, R = qr(A.T.conj())

        R = R.T.conj()

        gamma_inv_sqrt[...] = R[0, 0]
        gamma_inv_sqrt_k = R[1:, 0]
        Pch[...] = R[1:, 1:]

        k[...] = (gamma_inv_sqrt**-1) * gamma_inv_sqrt_k
        e[...] = d - (w_prev.conj() @ u)
        w[...] = w_prev + k * e.conj()

    return W[1:].T, E, K.T, gamma_inv_sqrt0, Pch0[-1]


# %%
def snir(y, x):
    return -10 * np.log10(np.mean((y - x) ** 2))


# %% [markdown]
# ### Theoretical Values

# %%
SNIR_raw = snir(y_m_max[-1], x_k)

# %% tags=["active-ipynb"]
SNIR_raw

# %%
SNIR_theory_raw = -10 * np.log10(4 * np.abs(ALPHA) ** 2 + sigma_v**2)

# %% tags=["active-ipynb"]
SNIR_theory_raw

# %%
SNIR_optimal = -10 * np.log10(sigma_v**2)

# %% tags=["active-ipynb"]
SNIR_optimal

# %% [markdown]
# ### RLS

# %%
W, E, K, P_final = rls(LAMBDA, DELTA, d, A)

w_f = W[:, -1]

# %% tags=["active-ipynb"]
w_f

# %%
x_est = einsum(w_f.conj(), y_m_max, "i, i j -> j")

# %%
SNIR_equalized = snir(x_est, x_k)

# %% tags=["active-ipynb"]
SNIR_equalized

# %% [markdown]
# ### Inverse QRD RLS

# %%
W_inv, E_inv, K_inv, gamma_inv, Pch_final = invqrdrls(LAMBDA, DELTA, d, A)

w_inv_f = W_inv[:, -1]

# %% tags=["active-ipynb"]
w_inv_f

# %%
x_est = einsum(w_inv_f.conj(), y_m_max, "i, i j -> j")

# %%
SNIR_inv_equalized = snir(x_est, x_k)

# %% tags=["active-ipynb"]
SNIR_inv_equalized

# %% tags=["active-ipynb"]
norm(P_final - (Pch_final @ Pch_final.T.conj()))
