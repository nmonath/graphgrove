# !/usr/bin/env python

from distutils.core import setup

setup(name='graphgrove',
      version='0.01',
      packages=['graphgrove'],
      install_requires=[
          "absl-py",
          "numpy"
      ],
      package_dir={'graphgrove': 'graphgrove'}
      )
