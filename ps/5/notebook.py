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

from numpy.linalg import norm

# %% [markdown]
# ## System Model Equations


# %%
def uhl_process(x, t, Q_cholesky=None, *, rng=None):
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
        uhl_process(x, t, Q_cholesky, rng=rng)
        if method == "e"
        else uhl_process(
            x + dt / 2 * uhl_process(x, t),
            t + dt / 2,
            Q_cholesky,
            rng=rng,
        )
    )

    x_update = x + dt * x_update

    return x_update


# %%
def uhl_measurement(x, t, Q_cholesky=None, *, rng=None):
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
N_x = 5

w = np.ones(2 * N_x + 1) / 3
w[1:] = (1 - w[1:]) / (2 * N_x)

sigma_points = np.concatenate(
    [
        np.sqrt(N_x / (1 - w[0])) * np.eye(N_x, N_x + 1, 1),
        -np.sqrt(N_x / (1 - w[0])) * np.eye(N_x),
    ],
    axis=1,
)

# %%
DT = 0.1
STEPS = 500
