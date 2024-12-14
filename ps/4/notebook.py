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

from numpy.linalg import (
    eigvals,
    matrix_rank,
)

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
N = A_1.shape[-1]

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
