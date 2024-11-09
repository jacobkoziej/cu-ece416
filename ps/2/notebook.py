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
# noteboook.py -- ps2: beamforming problems
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# %%
import matplotlib.pyplot as plt
import numpy as np

from einops import (
    einsum,
    rearrange,
    repeat,
)

# %% [markdown]
# ## Sensor Array Signal Model


# %%
def gen_a(aoa):
    assert aoa.shape[-1] == 2

    if aoa.ndim > 1:
        assert aoa.ndim == 2

    theta = aoa[..., 0]
    phi = aoa[..., 1]

    return np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    )


# %%
def gen_r(m, d):
    assert m.shape[-1] == 3

    return m * d


# %%
def gen_S(a, r, llambda):
    assert a.shape[-1] == 3
    assert r.shape[-1] == 3

    if a.ndim <= 2:
        a = rearrange(a, "... xyz -> ... 1 xyz")

    if r.ndim <= 1:
        r = rearrange(r, "xyz -> 1 xyz")

    S = -1j * np.pi * einsum(a, r, "... M i, ... M i -> ... M") / llambda
    S = np.exp(S) / np.sqrt(S.shape[-1])

    return S


# %%
def gen_A(rng, dB, N):
    scale = np.sqrt(10 ** (dB / 10))
    scale = repeat(scale, "... L -> ... N L", N=N)

    return (rng.normal(scale=scale) + 1j * rng.normal(scale=scale)) / np.sqrt(2)


# %%
def gen_V(rng, dB, N):
    M = dB.shape[-1]

    scale = np.sqrt((10 ** (dB / 10)) / M)
    scale = repeat(scale, "... M -> ... N M", N=N)

    return (rng.normal(scale=scale) + 1j * rng.normal(scale=scale)) / np.sqrt(2)


def gen_X(S, A, V):
    return einsum(S, A, "... L M, ... N L -> ... N M") + V


def theoretical_correlation(S, signal_dB, noise_dB):
    R_A = np.diag(np.sqrt(10 ** (signal_dB / 10)))
    R_V = np.diag(np.sqrt(10 ** (noise_dB / 10)))

    R_U = einsum(S.conj(), R_A, "... j i, ... j k -> ... i k")
    R_U = einsum(R_U, S, "... i j, ... j k -> ... i k")
    R_U += R_V

    return R_U


def estimated_correlation(X):
    N = X.shape[-2]

    return einsum(X, X.conj(), "... j i, ... j k -> ... i k") / N


# %% [markdown]
# ## SVD and MUSIC / MVDR Spectra

# %%
d = 0.5
llambda = 1
N = 100

# %%
m = np.arange(-10, 10 + 1)
m = repeat(m, "... m -> ... m 1")
m = np.append(
    np.pad(m, ((0, 0), (0, 2)), "constant", constant_values=0),
    np.pad(m, ((0, 0), (1, 1)), "constant", constant_values=0),
    axis=0,
)
m = np.unique(m, axis=0)

M = m.shape[-2]

# %%
signal_dB = np.array([0, -4, -8])
noise_dB = -12 * np.ones(M)

L = signal_dB.shape[-1]

# %%
aoa = np.array(
    [
        [15, 30],
        [20, 40],
        [30, -40],
    ]
)
aoa = np.deg2rad(aoa)

# %% [markdown]
# (a) Compute an SVD of $X$, draw a stem plot of the singular values,
# and confirm there are 3 dominant singular values. Compute $\sigma_4 /
# \sigma_3$ to see the "drop off" to the next singular value.

# %%
SEED = 0xBD4ED172
rng = np.random.default_rng(SEED)

# %%
r = gen_r(m, d)
a = gen_a(aoa)
S = gen_S(a, r, llambda)
A = gen_A(rng, signal_dB, N)
V = gen_V(rng, noise_dB, N)
X = gen_X(S, A, V)

# %%
P_H, svdvals, Q = np.linalg.svd(X)


# %%
def plot_svdvals(svdvals):
    ax = plt.subplot()

    ax.stem(svdvals)
    ax.set_xticks(range(len(svdvals)))
    ax.set_xticklabels([])
    ax.set_xlabel(r"$\sigma$")
    ax.set_title("Singular Values of $X$ in Descending Order")

    return ax


# %% tags=["active-ipynb"]
_ = plot_svdvals(svdvals)

# %%
drop_off = svdvals[3] / svdvals[2]

# %% tags=["active-ipynb"]
drop_off

# %% [markdown]
# (b) Working off the theoretical correlation matrix $R$, compute the
# eigenvalues (sorted in descending order), draw a stem plot, and confirm there are 3 dominant
# values. Compare $\lambda_4 / \lambda_3$.

# %%
R_theoretical = theoretical_correlation(S, signal_dB, noise_dB)


# %%
def get_eigvals(A):
    return np.flip(np.sort(np.linalg.eigvals(A)), axis=-1)


# %%
eigvals = get_eigvals(R_theoretical)


# %%
def plot_eigvals(eigvals):
    ax = plt.subplot()

    ax.stem(np.abs(eigvals))
    ax.set_xticks(range(len(eigvals)))
    ax.set_xticklabels([])
    ax.set_xlabel(r"$|\,\lambda|$")
    ax.set_title(r"Eigen Values of $R_{\text{theoretical}}$ in Descending Order")

    return ax


# %% tags=["active-ipynb"]
_ = plot_eigvals(eigvals)

# %%
theoretical_drop_off = eigvals[3] / eigvals[2]

# %% tags=["active-ipynb"]
theoretical_drop_off

# %% [markdown]
# (c) Using the SVD, compute the projection matrix $P_N$ onto the noise
# subspace. Compute $\left|P_N s(\Theta_\ell)\right|$, $1 \le \ell \le
# 3$, for the source singular vectors. These should be 0 in theory.

# %%
Q_L = Q[:L]

P_N = einsum(Q_L, Q_L.conj(), "j i, j k -> i k")
P_N = np.eye(*P_N.shape) - P_N

# %% tags=["active-ipynb"]
np.linalg.norm(einsum(P_N, S, "i j, ... k j -> ... i k"), axis=-1)

# %% [markdown]
# (d) We want to examine the MUSIC spectrum based on the SVD of the
# data, and the MVDR spectrum based on the estimated correlation matrix
# $\hat{R}$. This is comparable because both are based on the random
# data, not theoretical values. Generate a grid of $(\theta, \phi)$ with
# $0^{\circ} \le \theta \le 90^{\circ}$ and $−180^{\circ} \le \phi \le
# 180^{\circ}$. Obtain separate plots of each.

# %%
DEGREE_STEP = 5

theta = np.deg2rad(np.arange(0, 90, DEGREE_STEP))
phi = np.deg2rad(np.arange(-180, 180, DEGREE_STEP))

theta, phi = np.meshgrid(theta, phi)

grid_shape = theta.shape

theta = theta.flatten()
phi = phi.flatten()


# %%
def music(S, P_N):
    S_music = einsum(P_N, S, "... i j, ... k j -> ... i k")
    S_music = einsum(S.conj(), S_music, "... i j, ... j k -> ... i k")

    return 1 / S_music


# %%
def mvdr(S, R_inverse):
    return music(S, R_inverse)


# %%
aoa_surface = np.stack([theta, phi], axis=-1)
a_surface = gen_a(aoa_surface)
S_surface = gen_S(a_surface, r, llambda)
R_estimate = estimated_correlation(X)

# %%
S_music = np.diagonal(music(S_surface, P_N))
S_mvdr = np.diagonal(mvdr(S_surface, np.linalg.inv(R_estimate)))

# %%
S_music = S_music.reshape(grid_shape)
S_mvdr = S_mvdr.reshape(grid_shape)
theta = theta.reshape(grid_shape)
phi = phi.reshape(grid_shape)


# %%
def plot_contour(theta, phi, S, aoa, approach):
    ax = plt.subplot()

    ax.contour(theta, phi, np.abs(S))
    ax.scatter(aoa[..., 0], aoa[..., 1])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(r"$|S_\text{" + approach + r"}|$")

    return ax


# %% tags=["active-ipynb"]
_ = plot_contour(theta, phi, S_music, aoa, "MUSIC")

# %% tags=["active-ipynb"]
_ = plot_contour(theta, phi, S_mvdr, aoa, "MVDR")

# %% [markdown]
# (e) Take the "slice" where $\theta = 20^{\circ}$ and obtain the 1-D
# plot for $−180^{\circ} \le \phi \le 180^{\circ}$. You'll note one of
# your sources lands exactly on this slice.

# %%
theta_index = round((90 / DEGREE_STEP) * (30 / 180))

phi_theta = phi[:, theta_index]
S_music_theta = S_music[:, theta_index]
S_mvdr_theta = S_mvdr[:, theta_index]


# %%
def plot_slice(angle, S, approach, symbol, title):
    ax = plt.subplot()

    ax.plot(angle, np.abs(S))
    ax.set_xlabel(f"${symbol}$")
    ax.set_ylabel(r"$|S_\text{" + approach + r"}|$")
    ax.set_title(title)

    return ax


# %% tags=["active-ipynb"]
_ = plot_slice(phi_theta, S_music_theta, "MUSIC", r"\phi", r"$\theta = 30^{\circ}$")

# %% tags=["active-ipynb"]
_ = plot_slice(phi_theta, S_mvdr_theta, "MVDR", r"\phi", r"$\theta = 30^{\circ}$")

# %% [markdown]
# (f) Repeat for a slice at $\phi = 30^{\circ}$ and $0^{\circ} \le
# \theta \le 90^{\circ}$.

# %%
phi_index = round((360 / DEGREE_STEP) * ((180 + 30) / 360))

theta_phi = theta[phi_index]
S_music_phi = S_music[phi_index]
S_mvdr_phi = S_mvdr[phi_index]


# %% tags=["active-ipynb"]
_ = plot_slice(theta_phi, S_music_phi, "MUSIC", r"\theta", r"$\phi = 30^{\circ}$")

# %% tags=["active-ipynb"]
_ = plot_slice(theta_phi, S_mvdr_phi, "MVDR", r"\theta", r"$\phi = 30^{\circ}$")
