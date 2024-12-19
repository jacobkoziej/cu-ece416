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

# ruff: noqa: E402

# %%
import matplotlib.pyplot as plt
import numpy as np

from einops import repeat
from numpy.linalg import (
    inv,
    norm,
    svd,
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
        ]
    )


# %%
def gen_r(m, d):
    assert m.shape[-1] == 3

    return m * d


# %%
def gen_S(a, r, llambda):
    assert a.shape[-2] == 3
    assert r.shape[-1] == 3

    S = -1j * np.pi * (r @ a) / llambda
    M = S.shape[-2]
    S = np.exp(S) / np.sqrt(M)

    return S


# %%
def gen_A(rng, dB, N):
    scale = np.sqrt(10 ** (dB / 10))
    scale = repeat(scale, "... L -> ... L N", N=N)

    A = rng.normal(scale=scale) + 1j * rng.normal(scale=scale) / np.sqrt(2)

    return A


# %%
def gen_V(rng, dB, N):
    M = dB.shape[-1]

    scale = np.sqrt((10 ** (dB / 10)) / M)
    scale = repeat(scale, "... M -> ... M N", N=N)

    V = rng.normal(scale=scale) + 1j * rng.normal(scale=scale) / np.sqrt(2)

    return V


def gen_X(S, A, V):
    return (S @ A) + V


def theoretical_correlation(S, signal_dB, noise_dB):
    R_A = np.diag(np.sqrt(10 ** (signal_dB / 10)))
    R_V = np.diag(np.sqrt(10 ** (noise_dB / 10)))

    R_U = (S @ R_A @ S.conj().T) + R_V

    return R_U


def estimated_correlation(X):
    N = X.shape[-1]

    return (X @ X.conj().T) / N


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
Q, svdvals, P_H = svd(X)

# %% tags=["active-ipynb"]
plt.figure()
plt.stem(svdvals)
plt.xticks(range(len(svdvals)))
plt.xticks([])
plt.xlabel(r"$\sigma$")
plt.title("Singular Values of $X$ in Descending Order")
plt.show()

# %%
drop_off = svdvals[3] / svdvals[2]

# %% tags=["active-ipynb"]
drop_off

# %% [markdown]
# (b) Working off the theoretical correlation matrix $R$, compute the
# eigenvalues (sorted in descending order), draw a stem plot, and
# confirm there are 3 dominant values. Compare $\lambda_4 / \lambda_3$.

# %%
R_theoretical = theoretical_correlation(S, signal_dB, noise_dB)


# %%
def get_eigvals(A):
    return np.flip(np.sort(np.linalg.eigvals(A)), axis=-1)


# %%
eigvals = get_eigvals(R_theoretical)

# %% tags=["active-ipynb"]
plt.figure()
plt.stem(np.abs(eigvals))
plt.xticks(range(len(eigvals)), [])
plt.xlabel(r"$|\,\lambda|$")
plt.title(r"Eigen Values of $R_{\text{theoretical}}$ in Descending Order")
plt.show()

# %%
theoretical_drop_off = eigvals[3] / eigvals[2]

# %% tags=["active-ipynb"]
theoretical_drop_off

# %% [markdown]
# (c) Using the SVD, compute the projection matrix $P_N$ onto the noise
# subspace. Compute $\left|P_N s(\Theta_\ell)\right|$, $1 \le \ell \le
# 3$, for the source singular vectors. These should be 0 in theory.

# %%
Q_L = Q[:, :L]

P_N = Q_L @ Q_L.conj().T
P_N = np.eye(*P_N.shape) - P_N

# %% tags=["active-ipynb"]
norm(P_N @ S, axis=-1)

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
    return np.diagonal(1 / (S.conj().T @ P_N @ S))


# %%
def mvdr(S, R_inverse):
    return music(S, R_inverse)


# %%
aoa_surface = np.stack([theta, phi], axis=-1)
a_surface = gen_a(aoa_surface)
S_surface = gen_S(a_surface, r, llambda)
R_estimate = estimated_correlation(X)
R_estimate_inv = np.linalg.inv(R_estimate)

# %%
S_music = music(S_surface, P_N)
S_mvdr = mvdr(S_surface, R_estimate_inv)

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
plt.figure()
plot_slice(
    phi_theta,
    S_music_theta,
    "MUSIC",
    r"\phi",
    r"$\theta = 30^{\circ}$",
)
plt.show()

# %% tags=["active-ipynb"]
plt.figure()
plot_slice(phi_theta, S_mvdr_theta, "MVDR", r"\phi", r"$\theta = 30^{\circ}$")
plt.show()

# %% [markdown]
# (f) Repeat for a slice at $\phi = 30^{\circ}$ and $0^{\circ} \le
# \theta \le 90^{\circ}$.

# %%
phi_index = round((360 / DEGREE_STEP) * ((180 + 30) / 360))

theta_phi = theta[phi_index]
S_music_phi = S_music[phi_index]
S_mvdr_phi = S_mvdr[phi_index]


# %% tags=["active-ipynb"]
plt.figure()
plot_slice(theta_phi, S_music_phi, "MUSIC", r"\theta", r"$\phi = 30^{\circ}$")
plt.show()

# %% tags=["active-ipynb"]
plt.figure()
plot_slice(theta_phi, S_mvdr_phi, "MVDR", r"\theta", r"$\phi = 30^{\circ}$")
plt.show()

# %% [markdown]
# ## Optimal Beamforming: MVDR and GSC


# %%
def gen_w_q(C, g):
    return (C @ inv(C.conj().T @ C)) @ g


# %%
def gen_w_mvdr(S, R_inverse):
    return (R_inverse @ S) * mvdr(S, R_inverse)


# %%
def gen_w_gsc(C_a, R, w_q):
    w_gsc = C_a @ inv(C_a.conj().T @ R @ C_a) @ C_a.conj().T @ R
    w_gsc = (np.eye(*w_gsc.shape) - w_gsc) @ w_q

    return w_gsc


# %%
C = S
C_a, svdvals, _ = svd(C)
C_a = C_a[:, L:]

g = np.eye(L)
w_q = gen_w_q(C, g)

w_mvdr = gen_w_mvdr(S, R_estimate_inv)
w_gsc = gen_w_gsc(C_a, R_estimate, w_q)

# %%
S_mvdr = w_mvdr.conj().T @ S_surface
S_gsc = w_gsc.conj().T @ S_surface

# %%
S_mvdr = S_mvdr.reshape((-1,) + grid_shape)
S_gsc = S_gsc.reshape((-1,) + grid_shape)


# %%
def plot_contour(theta, phi, S, aoa, approach):
    fig, axs = plt.subplots(1, S.shape[0])

    for i, (s, ax) in enumerate(zip(S, axs)):
        ax.contour(theta, phi, np.abs(s))
        ax.scatter(aoa[..., 0], aoa[..., 1], c="red")
        ax.set_title(f"AOA = {i}")

    fig.supxlabel(r"$\theta$")
    fig.supylabel(r"$\phi$")
    fig.suptitle(f"{approach} Array Response")
    fig.tight_layout()

    return fig, ax


# %% tags=["active-ipynb"]
fig, _ = plot_contour(theta, phi, S_mvdr, aoa, "MVDR")
fig.show()


# %% tags=["active-ipynb"]
fig, _ = plot_contour(theta, phi, S_gsc, aoa, "GSC")
fig.show()

# %%
phi_theta = np.linspace(-np.pi, np.pi, 1_000)
aoa_theta = np.zeros((phi_theta.shape[0], 2))
aoa_theta[:, 0] = np.deg2rad(30)
aoa_theta[:, 1] = phi_theta
a_theta = gen_a(aoa_theta)
S_theta = gen_S(a_theta, r, llambda)

# %%
S_mvdr_theta = w_mvdr.conj().T @ S_theta
S_gsc_theta = w_gsc.conj().T @ S_theta


# %%
def plot_slice(angle, S, approach, symbol, title):
    fig, ax = plt.subplots(1, 1)

    for s in S:
        ax.plot(angle, 20 * np.log10(np.abs(s)))

    fig.supxlabel(f"${symbol}$")
    fig.supylabel(f"{approach} Array Response")
    fig.suptitle(title)
    fig.tight_layout()

    return fig, ax


# %% tags=["active-ipynb"]
fig, _ = plot_slice(
    phi_theta,
    S_mvdr_theta,
    "MVDR",
    r"\phi",
    r"$\theta = 30^{\circ}$",
)
fig.show()


# %% tags=["active-ipynb"]
fig, _ = plot_slice(
    phi_theta,
    S_gsc_theta,
    "GSC",
    r"\phi",
    r"$\theta = 30^{\circ}$",
)
fig.show()

# %% tags=["active-ipynb"]
np.rad2deg(aoa)  # theta, phi

# %% tags=["active-ipynb"]
20 * np.log10(np.abs(w_mvdr.conj().T @ S))
