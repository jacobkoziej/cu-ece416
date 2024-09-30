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

from typing import Final

from scipy.linalg import toeplitz
from scipy.optimize import fsolve
from scipy.signal import (
    sosfilt,
    zpk2sos,
)

# %%
p: Final[np.ndarray] = np.array(
    [
        [0.90, +0.80],
        [0.95, +0.80],
        [0.95, -0.90],
    ]
)

# %% [markdown]
# ## Task 1: Synthesize the Random Signal

# %%
p_max: Final[np.ndarray] = np.apply_along_axis(np.max, 1, np.abs(p))
N_init: Final[np.ndarray] = np.ceil(
    fsolve(lambda n, p: p**n - 0.01, np.ones(p_max.shape), p_max)
)

# %% tags=["active-ipynb"]
N_init

# %%
ITERATIONS: Final[int] = 50
SEED: Final[int] = 0x432F2AF7

M: Final[np.ndarray] = np.array([2, 4, 10])
N: Final[np.ndarray] = (N_init + ITERATIONS).astype(np.uint)

assert (ITERATIONS > M).all()

mu: Final[int] = 0
sigma: Final[int] = 1

# %%
rng: np.random.Generator = np.random.default_rng(SEED)

v: Final[list[np.ndarray]] = [rng.normal(mu, sigma, n) for n in N]
sos: Final[list[np.ndarray]] = [zpk2sos(np.zeros(p.shape), p, 1) for p in p]

x: list[np.ndarray] = [sosfilt(sos, v) for (sos, v) in zip(sos, v)]
x: Final[list[np.ndarray]] = [x[-ITERATIONS:] for x in x]

# %%
X: Final[list[np.ndarray]] = [
    toeplitz(np.flip(x[:m]).T, x[(m - 1) :]) for (x, m) in zip(x, M)
]
