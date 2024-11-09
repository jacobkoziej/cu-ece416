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

    if a.ndim == 1:
        a = rearrange(a, "i -> 1 i")
    if r.ndim == 1:
        r = rearrange(r, "i -> 1 i")

    assert a.ndim <= 3
    assert r.ndim <= 3

    assert a.shape == r.shape

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
