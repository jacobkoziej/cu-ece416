# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# ruff: noqa: E402
# ruff: noqa: F821

EnsureSConsVersion(4, 7, 0)
EnsurePythonVersion(3, 12)


import os


env = Environment(
    ENV={
        "PATH": os.environ["PATH"],
        "TERM": os.environ.get("TERM"),
    },
    tools=[
        "default",
        "github.jacobkoziej.scons-tools.Jupyter.NbConvert",
        "github.jacobkoziej.scons-tools.Jupytext",
        "github.jacobkoziej.scons-tools.Papermill",
    ],
)
env.AppendUnique(
    NBCONVERTFLAGS=[
        "--TagRemovePreprocessor.enabled=True",
        "--TagRemovePreprocessor.remove_cell_tags=\"{'parameters'}\"",
        "--to=html",
    ],
)

build = "build"


ps = SConscript(
    "ps/SConscript.py",
    exports=[
        "env",
    ],
    variant_dir=f"{build}/ps",
)
