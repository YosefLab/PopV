#!/usr/bin/env python
"""Shim to allow Github to detect the package, build is done with hatch."""

# !/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup(name="popv")
