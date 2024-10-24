# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# ruff: noqa: F821

from pathlib import Path

Import("env")

p_n = [
    {
        "p_0": 0.9,
        "p_1": 0.8,
    },
    {
        "p_0": 0.95,
        "p_1": 0.8,
    },
    {
        "p_0": 0.95,
        "p_1": -0.9,
    },
]
M = [2, 4, 10]

notebooks = []

notebook_raw = env.Jupytext("notebook-raw.ipynb", "notebook.py")

for m in M:
    for p in p_n:
        extension = []
        flags = []

        for k, v in p.items():
            extension += [f"{k}_{v}"]
            flags += [f"--parameters {k} {v}"]

        extension += [
            f"M_{m}",
        ]
        flags += [
            f"--parameters M {m}",
        ]

        extension = "__".join(extension)
        flags = " ".join(flags)

        notebooks += env.Papermill(
            f"notebook__{extension}.ipynb",
            notebook_raw,
            PAPERMILLFLAGS=flags,
        )

ps1 = []

for notebook in notebooks:
    ps1 += env.NbConvert(
        str(Path(str(notebook)).with_suffix(".html")),
        notebook,
    )

Return("ps1")
