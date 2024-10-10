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
# noteboook.py -- ps1: initial experimentation
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# %% tags=["parameters"]
ITERATIONS = None
M = None
p_0 = None
p_1 = None

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy

from scipy.fft import (
    fft,
    fftshift,
)
from scipy.linalg import toeplitz
from scipy.optimize import fsolve
from scipy.signal import (
    sosfilt,
    zpk2sos,
)

# %%
p = np.array([p_0, p_1])

# %% [markdown]
# ## Task 1: Synthesize the Random Signal

# %%
p_max = np.max(np.abs(p))
N_init = int(np.ceil(fsolve(lambda n, p: p**n - 0.01, 1, p_max)).item())

# %% tags=["active-ipynb"]
N_init

# %%
SEED = 0x432F2AF7

N_0 = ITERATIONS + M - 1
N = N_init + N_0

mu = 0
sigma = np.sqrt(1)

# %%
rng = np.random.default_rng(SEED)

v = rng.normal(mu, sigma, N)
sos = zpk2sos([0], p, 1)

x = sosfilt(sos, v)
x = x[-N_0:]

# %%
X = toeplitz(np.flip(x[:M]), x[(M - 1) :])

# %% [markdown]
# ## Task 2: Analyze the Random Signal


# %%
def gen_r(p, m):
    K = np.ones(p.shape)

    K[0] *= p[0] * (1 + p[1] ** 2) - p[1] * (1 + p[0] ** 2)
    K[0] *= 1 - p[0] ** 2
    K[0] **= -1
    K[0] *= p[0]

    K[1] *= p[1] * (1 + p[0] ** 2) - p[0] * (1 + p[1] ** 2)
    K[1] *= 1 - p[1] ** 2
    K[1] **= -1
    K[1] *= p[1]

    r = p ** np.abs(m.T[:, np.newaxis])

    r *= K

    r = np.sum(r, axis=-1)

    return r


# %%
L = 20

m = np.arange(L * 100)
r = gen_r(p, m)
R = toeplitz(r[:L])

# %% [markdown]
# 1. Compute the eigenvalues of $R_{L+1}$, verify they are positive
# real, and stem plot them in descending order.


# %%
def get_eig(A: np.ndarray):
    return np.flip(np.sort(np.linalg.eig(A).eigenvalues), axis=-1)


# %%
eig = get_eig(R)

assert (eig >= 0).all()


# %%
def plot_eig(eig):
    ax = plt.subplot()

    ax.stem(eig)
    ax.set_xticks(range(len(eig)))
    ax.set_xlabel(r"$\lambda$")
    ax.set_title("Eigen Values of $R_{L + 1}$ in Descending Order")

    return ax


# %% tags=["active-ipynb"]
ax = plot_eig(eig)

# %% [markdown]
# 2. For the chain of submatrices, $1 \le \ell \le L + 1$, compute the
# sequence of $\lambda_{\text{min}}$ and $\lambda_{\text{max}}$.

# %%
eig = [eig]

eig += [get_eig(R[:l, :l]) for l in range(eig[0].shape[-1], 0, -1)]

eig_min = np.flip(np.array([eig[-1] for eig in eig]).T, axis=-1)
eig_max = np.flip(np.array([eig[0] for eig in eig]).T, axis=-1)


# %%
def plot_eig(eig, dir):
    ax = plt.subplot()

    ax.stem(eig)
    ax.set_xticks(np.arange(eig.shape[-1]))
    ax.set_xlabel(r"$L - 1$")
    ax.set_title(r"$\lambda_{\text{" + dir + r"}}$ for $R_{L}$")

    return ax


# %% tags=["active-ipynb"]
ax = plot_eig(eig_min, "min")

# %% tags=["active-ipynb"]
ax = plot_eig(eig_max, "max")

# %% [markdown]
# 3. Compute and plot $S(\omega)$. Also determine $S_{\text{min}}$,
# $S_{\text{max}}$ and verify that in all cases $S_{\text{min}} \le
# \lambda_{\text{min}}$ and $\lambda_{\text{max}} \le S_{\text{max}}$.

# %%
S = fftshift(fft(r))


# %%
def plot_S(S):
    ax = plt.subplot()

    omega = np.linspace(-np.pi, np.pi, len(S))

    ax.plot(omega, 10 * np.log10(np.abs(S)))

    ticks = [
        r"$-\pi$",
        r"$-3\pi / 4$",
        r"$-\pi / 2$",
        r"$-\pi / 4$",
        "0",
        r"$\pi / 4$",
        r"$\pi / 2$",
        r"$3\pi / 4$",
        r"$\pi$",
    ]

    ax.set_xticks(np.linspace(omega[0], omega[-1], len(ticks)), ticks)
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel(r"$\left|S(\omega)\right|$ [dB]")

    return ax


# %% tags=["active-ipynb"]
ax = plot_S(S)

# %%
assert (S.min(-1) <= eig[-1].min()).all()
assert (eig[-1].max(-1) <= S.max()).all()

# %% [markdown]
# 4. Now estimate $R$ with $X$.

# %%
R_M = R[:M, :M]

K = ITERATIONS

R_hat = (X @ X.T.conj()) / K

# %% tags=["active-ipynb"]
R_hat

# %% [markdown]
# It appears that $\hat{R}$ is **not** exactly Toeplitz.


# %%
def plot_r_vs_r_hat(r, r_hat):
    ax = plt.subplot()

    ax.stem(r)
    ax.stem(r_hat, linefmt="g:")
    ax.legend(["$r$", r"$\hat{r}$"])
    ax.set_xticks(np.arange(len(r)))
    ax.set_xlabel("m")
    ax.set_ylabel("r")
    ax.set_title("Theoretical vs Estimate")

    lim = [r, r_hat]
    lim = [np.min(lim) - 2, np.max(lim) + 2]

    ax.set_ylim(lim)

    return ax


# %% tags=["active-ipynb"]
r_hat = R_hat[0, :]

ax = plot_r_vs_r_hat(r[: len(r_hat)], r_hat)

# %% [markdown]
# 5. Suppose we compute the singular values of $X$; they provide the
# same information as the eigenvalues of $\hat{R} = \frac{1}{K} X
# X^{\text{T}}$. How are they related? Viewed another way, if we want to
# perform SVD analysis, it should be done on a scaled matrix $\alpha X$.
# What is the appropriate choice for $\alpha$?

# %% [markdown]
# We know that $\lambda \approx \frac{1}{N} \sigma^2$ and $X = U \Sigma
# V^{\text{H}}$, so it follows that $\alpha$ is equal to the square root
# of the number of columns of $X$.

# %% [markdown]
# 6. As another check, approximate $r[0]$ with your computed PSD.

# %%
r_0_approx = np.sum(S * (2 * np.pi / len(S))) / (2 * np.pi)

# %% tags=["active-ipynb"]
r[0] - r_0_approx
