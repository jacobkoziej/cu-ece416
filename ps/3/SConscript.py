# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# ruff: noqa: F821

from pathlib import Path

Import("env")

notebook_raw = env.Jupytext("notebook-raw.ipynb", "notebook.py")[0]

notebook = env.Papermill("notebook.ipynb", notebook_raw)[0]

ps3 = env.NbConvert(str(Path(str(notebook)).with_suffix(".html")), notebook)

Return("ps3")
