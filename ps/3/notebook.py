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
