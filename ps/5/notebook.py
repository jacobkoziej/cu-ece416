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
# noteboook.py -- ps5: unscented kalman filter
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# ruff: noqa: E402

# %%
import numpy as np

from collections import namedtuple

from numpy.linalg import (
    cholesky,
    inv,
    norm,
)

# %% [markdown]
# ## System Model Equations


# %%
def uhl_process(x, Q_cholesky=None, *, rng=None):
    beta_0 = 0.597983

    G_m_0 = 3.986e5
    R_0 = 6374
    H_0 = 13.406

    if Q_cholesky is None:
        Q_cholesky = np.diag(np.sqrt([0, 0, 2.4064e-5, 2.4064e-5, 0]))

    beta = beta_0 * np.exp(x[4])

    R = norm(x[:2])
    V = norm(x[2:4])
    D = -beta * V * np.exp((R_0 - R) / H_0)
    G = np.array([-G_m_0 / R**3])

    if rng is None:
        rng = np.random.default_rng()

    N = x.shape[0]

    v = Q_cholesky @ rng.normal(0, 1, (N, 1))

    x_dot = (
        np.array([x[2], x[3], D * x[2] + G * x[0], D * x[3] + G * x[1], [0]])
        + v
    )

    return x_dot


# %%
def uhl_process_simulation(x, t, dt, method="e", Q_cholesky=None, *, rng=None):
    assert method == "e" or method == "m"

    x_update = (
        uhl_process(x, Q_cholesky, rng=rng)
        if method == "e"
        else uhl_process(
            x + dt / 2 * uhl_process(x, np.zeros((x.shape[0], x.shape[0]))),
            Q_cholesky,
            rng=rng,
        )
    )

    x_update = x + dt * x_update

    return x_update


# %%
def uhl_measurement(x, Q_cholesky=None, *, rng=None):
    RADAR_CENTER = np.array([[6375, 0]]).T

    if Q_cholesky is None:
        Q_cholesky = np.diag(np.sqrt([1, 17e-3]))

    if rng is None:
        rng = np.random.default_rng()

    y = np.array(
        [
            np.array([norm(x[:2] - RADAR_CENTER)]),
            np.arctan2(x[1] - RADAR_CENTER[1], x[0] - RADAR_CENTER[0]),
        ]
    ) + Q_cholesky @ rng.normal(0, 1, (2, 1))

    return y


# %% [markdown]
# ## Theory and Setup


# %%
def gen_sigma_points(mu, C_cholesky):
    N_x = C_cholesky.shape[-1]

    assert mu.shape == (N_x, 1)
    assert C_cholesky.shape == (N_x, N_x)

    w = np.ones(2 * N_x + 1) / 3
    w[1:] = (1 - w[1:]) / (2 * N_x)

    sigma_points = np.concatenate(
        [
            np.sqrt(N_x / (1 - w[0])) * np.eye(N_x, N_x + 1, 1),
            -np.sqrt(N_x / (1 - w[0])) * np.eye(N_x),
        ],
        axis=1,
    )

    sigma_points = mu + (C_cholesky @ sigma_points)

    return w, sigma_points


# %%
def gen_sigma_mean(w, sigma_points):
    assert w.shape[-1] == sigma_points.shape[-1]
    assert sigma_points.ndim >= 2

    return np.sum(w * sigma_points, axis=1).reshape(-1, 1)


# %%
def gen_sigma_covariance(w, sigma_points_x, sigma_points_y=None):
    if sigma_points_y is None:
        sigma_points_y = sigma_points_x

    assert sigma_points_x.shape[-1] == sigma_points_y.shape[-1]
    assert w.shape[-1] == sigma_points_x.shape[-1]

    mu_x = gen_sigma_mean(w, sigma_points_x)
    mu_y = gen_sigma_mean(w, sigma_points_y)

    return (w * (sigma_points_x - mu_x)) @ (sigma_points_y - mu_y).T


# %% [markdown]
# Show that the empirical mean of the proposed sigma points is 0, the
# empirical covariance is $I$, the empirical skewness of each component
# is 0, and the empirical kurtosis of each component is $N_x / (1 -
# \omega_0)$. Evaluate this and compare to the kurtosis of Gaussian and
# comment. Show that selecting $w_0$ to get the kurtosis of 3 would
# require $w_0 < 0$, which is problematic. This set of sigma points does
# not match Gaussian kurtosis, but avoids negative weights.

# %%
N_x = 5

w, sigma_points = gen_sigma_points(np.zeros((N_x, 1)), np.eye(N_x))

# %%
mu_sigma_points = gen_sigma_mean(w, sigma_points)

# %% tags=["active-ipynb"]
mu_sigma_points

# %%
C_sigma_points = gen_sigma_covariance(w, sigma_points)

# %% tags=["active-ipynb"]
C_sigma_points

# %%
skew_sigma_points = np.mean((sigma_points - mu_sigma_points) ** 3, axis=1)
skew_sigma_points = skew_sigma_points / np.diagonal(C_sigma_points) ** 3

# %% tags=["active-ipynb"]
skew_sigma_points

# %%
kurtosis_sigma_points = np.sum(w * (sigma_points**4), axis=1)

# %% tags=["active-ipynb"]
kurtosis_sigma_points

# %% tags=["active-ipynb"]
1 / (2 * w[1])

# %%
kurtosis_sigma_points_excess = kurtosis_sigma_points - 3

# %% tags=["active-ipynb"]
kurtosis_sigma_points_excess

# %% [markdown]
# We have excess kurtosis which suggests the probability of a large
# deviation from the mean is greater than for the Gaussian distribution
# with the same variance.

# %%
w_0_gaussian_kurtosis = 1 - (N_x / 3)

assert w_0_gaussian_kurtosis < 0

# %% tags=["active-ipynb"]
w_0_gaussian_kurtosis

# %%
UkfState = namedtuple(
    "UkfState",
    ["Q_x", "Q_x_cholesky", "Q_y", "Q_y_cholesky", "measurement", "process"],
)
UkfOutput = namedtuple(
    "UkfOutput",
    ["K_prediction", "K_estimate", "x_prediction", "x_estimate"],
)


# %%
def ukf(state, K_prediction, x_prediction, y, n):
    w, sigma_points_x_prediction = gen_sigma_points(
        x_prediction, cholesky(K_prediction)
    )

    sigma_points_y_prediction = np.zeros(
        (y.shape[-2], sigma_points_x_prediction.shape[-1])
    )

    for i in range(sigma_points_x_prediction.shape[-1]):
        sigma_points_y_prediction[:, i : i + 1] = state.measurement(
            sigma_points_x_prediction[:, i : i + 1],
            np.zeros(state.Q_y_cholesky.shape),
        )

    y_prediction = gen_sigma_mean(w, sigma_points_y_prediction)

    K_xy_prediction = gen_sigma_covariance(
        w, sigma_points_x_prediction, sigma_points_y_prediction
    )
    K_yy_prediction = (
        gen_sigma_covariance(w, sigma_points_y_prediction) + state.Q_y
    )

    alpha = y - y_prediction
    G = K_xy_prediction @ inv(K_yy_prediction)
    x_estimate = x_prediction + (G @ alpha)
    K_estimate = K_prediction - (G @ K_yy_prediction @ G.T)

    w, sigma_points_x_estimate = gen_sigma_points(
        x_estimate, cholesky(K_estimate)
    )

    sigma_points_x_prediction = np.zeros(sigma_points_x_estimate.shape)

    for i in range(sigma_points_x_estimate.shape[-1]):
        sigma_points_x_prediction[:, i : i + 1] = state.process(
            sigma_points_x_estimate[:, i : i + 1], n, state.Q_x_cholesky
        )

    x_prediction = gen_sigma_mean(w, sigma_points_x_prediction)
    K_prediction = (
        gen_sigma_covariance(w, sigma_points_x_prediction) + state.Q_x
    )

    return UkfOutput(K_prediction, K_estimate, x_prediction, x_estimate)


# %% [markdown]
# ## Initial Conditions

# %%
Q_x = np.diag([10e-8, 10e-8, 2.404e-5, 2.404e-5, 10e-8])
Q_y = np.diag([1, 10e-3])

# %%
x = [np.array([[6400.4, 349.14, -1.8093, -6.7967, 0.6932]]).T]

x_prediction = [np.array([[6400, 350, -2, -7, 0.65]]).T]
x_estimate = []

K_prediction = [np.diag([10e-4, 10e-4, 10e-4, 10e-4, 1])]
K_estimate = []

# %%
DT = 0.1
STEPS = 500
