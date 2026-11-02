#!/usr/bin/env python

# Copyright (c) 2026 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Blackboard helpers used to pass one-shot signals between behaviours.

py_trees raises KeyError when reading a variable that was never written, while
the behaviours here read optional signals every tick, so reads return a default.
"""

import py_trees


def get_blackboard(name, default=None):
    """Returns the value stored under 'name', or 'default' if it is unset"""
    if not py_trees.blackboard.Blackboard.exists(name):
        return default
    return py_trees.blackboard.Blackboard.get(name)


def set_blackboard(name, value):
    """Stores 'value' under 'name', overwriting whatever was there"""
    py_trees.blackboard.Blackboard.set(name, value)
