# !/usr/bin/env python

# Copyright (c) 2021 GraphGrove
#
# Modified from CoverTree: https://github.com/manzilzaheer/CoverTree
# Copyright (c) 2017 Manzil Zaheer All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import sys

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        if type(__builtins__) is dict:
            __builtins__['__NUMPY_SETUP__'] = False
        else:
            __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


sgtreec_module = Extension('sgtreec',
        sources = ['src/sg_tree/sgtreecmodule.cxx', 'src/sg_tree/utils.cpp',  'src/sg_tree/sg_tree.cpp'],
        include_dirs=['lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'], #, '-DPRINTVER'], #, '-D_FLOAT64_VER_'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14'] #,'-DPRINTVER'] #, '-D_FLOAT64_VER_'], #'-D_FLOAT64_VER_', '-DPRINTVER'
)

covertreec_module = Extension('covertreec',
        sources = ['src/cover_tree/covertreecmodule.cxx', 'src/cover_tree/utils.cpp',  'src/cover_tree/cover_tree.cpp'],
        include_dirs=['lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'], #, '-DPRINTVER'], #, '-D_FLOAT64_VER_'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14'] #,'-DPRINTVER'] #, '-D_FLOAT64_VER_'], #'-D_FLOAT64_VER_', '-DPRINTVER'
)

scc_module = Extension('sccc',
        sources = ['src/scc/scccmodule.cxx',  'src/scc/scc.cpp'],
        include_dirs=['lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14'] #'-DPRINTVER']
)

llamac_module = Extension('llamac',
        sources = ['src/llama/llamacmodule.cxx',  'src/llama/llama.cpp'],
        include_dirs=['lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'], #, '-DPRINTVER'], #, '-D_FLOAT64_VER_'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14'] #,'-DPRINTVER'] #, '-D_FLOAT64_VER_'], #'-D_FLOAT64_VER_', '-DPRINTVER'
)

setuptools.setup(
    name="graphgrove",
    version="0.0.9",
    author="Nicholas Monath",
    author_email="nmonath@cs.umass.edu",
    description="Building flat, tree, and DAG structured clusterings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nmonath/graphgrove",
    project_urls={
        "Bug Tracker": "https://github.com/nmonath/graphgrove/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    package_dir={'graphgrove': 'graphgrove'},
    packages=['graphgrove'],
    cmdclass={'build_ext': build_ext},
    install_requires=['numpy>=1.21', 'scipy>=0.17', 'tqdm', 'absl-py'],
    ext_modules = [sgtreec_module, covertreec_module, scc_module, llamac_module],
    python_requires=">=3.6",
)
