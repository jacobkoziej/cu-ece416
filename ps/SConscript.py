# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

# ruff: noqa: F821

Import("env")

ps = []

for i in range(1, 6):
    ps += SConscript(
        f"{i}/SConscript.py",
        exports=[
            "env",
        ],
    )

Return("ps")
