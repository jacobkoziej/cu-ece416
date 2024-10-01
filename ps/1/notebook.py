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

# %%
import numpy as np
import scipy

from scipy.linalg import toeplitz
from scipy.optimize import fsolve
from scipy.signal import (
    correlate,
    sosfilt,
    zpk2sos,
)

# %%
p = np.array(
    [
        [0.90, +0.80],
        [0.95, +0.80],
        [0.95, -0.90],
    ]
)

# %% [markdown]
# ## Task 1: Synthesize the Random Signal

# %%
p_max = np.apply_along_axis(np.max, 1, np.abs(p))
N_init = np.ceil(fsolve(lambda n, p: p**n - 0.01, np.ones(p_max.shape), p_max))

# %% tags=["active-ipynb"]
N_init

# %%
ITERATIONS = 30
SEED = 0x432F2AF7

M = np.array([2, 4, 10])
N_0 = (ITERATIONS + M).astype(np.int64)
N = (N_init + N_0).astype(np.int64)

mu = 0
sigma = np.sqrt(1)

# %%
rng = np.random.default_rng(SEED)

v = [rng.normal(mu, sigma, n) for n in N]
sos = [zpk2sos([0], p, 1) for p in p]

x = [sosfilt(sos, v) for (sos, v) in zip(sos, v)]
x = [x[-n_0:] for (x, n_0) in zip(x, N_0)]

# %%
X = [[toeplitz(np.flip(x[:m]), x[(m - 1) :]) for m in M] for x in x]
