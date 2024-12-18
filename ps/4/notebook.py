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
# noteboook.py -- ps4: kalman filters
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# ruff: noqa: E402

# %%
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

from einops import repeat
from numpy.linalg import (
    eigvals,
    inv,
    matrix_rank,
    norm,
)
from scipy.linalg import solve_discrete_are

# %% [markdown]
# ## Preliminary Analysis

# %%
A_2 = np.array(
    [
        [-3, -5],
        [1, 1],
    ]
)
A_1 = A_2 / 2

C = np.array([[1, 0]])

Q_x = np.array(
    [
        [0.1, 0],
        [0, 0.1],
    ]
)
Q_y = np.array(0.05)

# %%
A_1_eig = eigvals(A_1)

# %% tags=["active-ipynb"]
A_1_eig

# %%
A_2_eig = eigvals(A_2)

# %% tags=["active-ipynb"]
A_2_eig

# %% [markdown]
# $A_1$ is the stable system since the magnitude of its eigen values are
# all less than 1.

# %%
p = C.shape[0]
N = A_1.shape[0]

# %%
O_1 = np.concatenate(
    [
        C @ np.eye(N),
        C @ A_1,
    ]
)
O_2 = np.concatenate(
    [
        C @ np.eye(N),
        C @ A_2,
    ]
)

assert matrix_rank(O_1) == N
assert matrix_rank(O_2) == N

# %% [markdown]
# ## Kalman Filter

# %%
ITERATIONS = 100
ITERATIONS_STABLE = 50

# %%
mu = 0
sigma_x = np.sqrt(Q_x[0, 0])
sigma_y = np.sqrt(Q_y)

x_0 = np.array([[1, 0]]).T

# %%
SEED = 0xCEAD8773

rng = np.random.default_rng(SEED)

# %%
x = [x_0]
y = []

for i in range(ITERATIONS_STABLE):
    x += [A_1 @ x[i] + rng.normal(mu, sigma_x)]
    y += [C @ x[i] + rng.normal(mu, sigma_y)]

for i in range(ITERATIONS_STABLE, ITERATIONS):
    x += [A_2 @ x[i] + rng.normal(mu, sigma_x)]
    y += [C @ x[i] + rng.normal(mu, sigma_y)]

# %%
gamma = 0.1

# %%
KalmanState = namedtuple("KalmanState", ["A", "C", "Q_x", "Q_y"])
KalmanOutput = namedtuple(
    "KalmanOutput",
    ["K_prediction", "K_estimate", "x_prediction", "x_estimate"],
)


# %%
def kalman_filter(state, K_prediction, x_prediction, y):
    R = inv(state.C @ K_prediction @ state.C.T.conj() + state.Q_y)
    G = K_prediction @ state.C.T.conj() @ R

    alpha = y - state.C @ x_prediction

    x_estimate = x_prediction + G @ alpha

    x_prediction = state.A @ x_estimate

    K_estimate = K_prediction - G @ state.C @ K_prediction

    K_prediction = state.A @ K_estimate @ state.A.T.conj() + state.Q_x

    return KalmanOutput(K_prediction, K_estimate, x_prediction, x_estimate)


# %%
K_prediction = [gamma * np.identity(N)]
K_estimate = []
x_prediction = [np.array([[0, 0]]).T]
x_estimate = []

state = KalmanState(A=A_1, C=C, Q_x=Q_x, Q_y=Q_y)

for i in range(ITERATIONS_STABLE):
    out = kalman_filter(state, K_prediction[i], x_prediction[i], y[i])

    K_prediction += [out.K_prediction]
    K_estimate += [out.K_estimate]
    x_prediction += [out.x_prediction]
    x_estimate += [out.x_estimate]

state = KalmanState(A=A_2, C=C, Q_x=Q_x, Q_y=Q_y)

for i in range(ITERATIONS_STABLE, ITERATIONS):
    out = kalman_filter(state, K_prediction[i], x_prediction[i], y[i])

    K_prediction += [out.K_prediction]
    K_estimate += [out.K_estimate]
    x_prediction += [out.x_prediction]
    x_estimate += [out.x_estimate]

# %%
K_ideal_1 = solve_discrete_are(
    a=A_1.T.conj(),
    b=C.T.conj(),
    q=Q_x,
    r=Q_y,
    e=np.identity(N),
    s=np.zeros((N, p)),
)
K_ideal_2 = solve_discrete_are(
    a=A_2.T.conj(),
    b=C.T.conj(),
    q=Q_x,
    r=Q_y,
    e=np.identity(N),
    s=np.zeros((N, p)),
)

# %% [markdown]
# a) For $1 \le n \le 100$, compute $||K(n,\ n - 1) - K_\text{ideal}||$
# and plot it.

# %%
K_prediction = np.concatenate(
    [
        np.stack(K_prediction[:ITERATIONS_STABLE]),
        np.stack(K_prediction[ITERATIONS_STABLE:ITERATIONS]),
    ]
)
K_ideal = np.concatenate(
    [
        repeat(K_ideal_1, "... -> N ...", N=ITERATIONS_STABLE),
        repeat(K_ideal_2, "... -> N ...", N=ITERATIONS - ITERATIONS_STABLE),
    ]
)

# %%
plt.figure()
plt.plot(norm(K_prediction - K_ideal, axis=(-2, -1)))
plt.axvline(ITERATIONS_STABLE, color="r", linestyle="--")
plt.text(
    ITERATIONS_STABLE - 5, 1.5, "Instability Added", color="red", rotation=90
)
plt.title(r"$||K(n,\ n - 1) - K_\text{ideal}||$")
plt.xlabel("n")
plt.show()

# %% [markdown]
# b) Plot the trajectories

# %%
x = np.stack(x[:-1])
x_prediction = np.stack(x_prediction[:-1])
x_estimate = np.stack(x_estimate)

# %%
plt.figure()
plt.plot(x[:, 0], x[:, 1])
plt.plot(x_prediction[:, 0], x_prediction[:, 1])
plt.plot(x_estimate[:, 0], x_estimate[:, 1])
plt.legend([r"$x$", r"$x_\text{prediction}$", r"$x_\text{estimate}$"])
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title(r"$x_1$ vs $x_2$")
plt.show()

# %% [markdown]
# c) Plot $x$ for $n > 50$ along with $||x - x_\text{prediction}||$ and
# $||x - x_\text{estimate}||$ for all $n$.

# %%
plt.figure()
plt.plot(x[ITERATIONS_STABLE:, 0], x[ITERATIONS_STABLE:, 1])
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title(r"$x_1$ vs $x_2$ for $51 \leq n \leq 100$")
plt.show()

# %%
plt.figure()
plt.plot(norm(x - x_prediction, axis=-2))
plt.xlabel("n")
plt.title(r"$||x - x_\text{prediction}||$")
plt.show()

# %%
plt.figure()
plt.plot(norm(x - x_estimate, axis=-2))
plt.xlabel("n")
plt.title(r"$||x - x_\text{estimate}||$")
plt.show()

# %% [markdown]
# d) Compute $\hat{K}_p$ and $\hat{K}_e$ before and after $n = 50$. Also
# show that $\Delta K = K_p - K_e > 0$.


# %%
def compute_error(x, x_hat):
    return (x - x_hat).squeeze().T


def compute_K_hat(e):
    M = e.shape[-1]

    return (e @ e.T) / M


# %%
e_p_1 = compute_error(x[:ITERATIONS_STABLE], x_prediction[:ITERATIONS_STABLE])
e_p_2 = compute_error(x[ITERATIONS_STABLE:], x_prediction[ITERATIONS_STABLE:])
e_e_1 = compute_error(x[:ITERATIONS_STABLE], x_estimate[:ITERATIONS_STABLE])
e_e_2 = compute_error(x[ITERATIONS_STABLE:], x_estimate[ITERATIONS_STABLE:])

K_hat_p_1 = compute_K_hat(e_p_1)
K_hat_p_2 = compute_K_hat(e_p_2)
K_hat_e_1 = compute_K_hat(e_e_1)
K_hat_e_2 = compute_K_hat(e_e_2)

# %% tags=["active-ipynb"]
norm(K_ideal_1 - K_hat_p_1)

# %% tags=["active-ipynb"]
norm(K_ideal_2 - K_hat_p_2)

# %% tags=["active-ipynb"]
norm(K_ideal_1 - K_hat_e_1)

# %% tags=["active-ipynb"]
norm(K_ideal_2 - K_hat_e_2)

# %%
K_p_1_final = K_prediction[ITERATIONS_STABLE - 1]
K_p_2_final = K_prediction[ITERATIONS - 1]
K_e_1_final = K_estimate[ITERATIONS_STABLE - 1]
K_e_2_final = K_estimate[ITERATIONS - 1]

# %% tags=["active-ipynb"]
eigvals(K_p_1_final - K_e_1_final)

# %% tags=["active-ipynb"]
eigvals(K_p_2_final - K_e_2_final)
