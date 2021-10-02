"""
Copyright (c) 2021 The authors of SCC All rights reserved.

Initially modified from CoverTree
https://github.com/manzilzaheer/CoverTree
Copyright (c) 2017 Manzil Zaheer All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
import sys

import numpy as np

from graphgrove.vec_scc import Cosine_SCC
from graphgrove.graph_builder import unit_norm

gt = time.time

np.random.seed(123)
cores = 1

print('======== Building Dataset ==========')
N=100
K=5
D=784
means = 20*np.random.rand(K,D) - 10
x = np.vstack([np.random.randn(N,D) + means[i] for i in range(K)])
np.random.shuffle(x)
x = unit_norm(x)
x = x.astype(np.float32)
x = np.require(x, requirements=['A', 'C', 'O', 'W'])
y = np.vstack([np.random.randn(N,D) + means[i] for i in range(K)])
y = y.astype(np.float32)
y = np.require(y, requirements=['A', 'C', 'O', 'W'])

print('======== SCC ==========')
t = gt()
num_rounds = 50
thresholds = np.geomspace(1.0, 0.001, num_rounds).astype(np.float32)
scc = Cosine_SCC(k=5, num_rounds=num_rounds, thresholds=thresholds, index_name='cosine_sgtree', cores=cores, verbosity=1)
scc.partial_fit(x)
b_t = gt() - t
print("Clustering time:", b_t, "seconds")
sys.stdout.flush()

print('======== MB-SCC ==========')
t = gt()
num_rounds = 50
thresholds = np.geomspace(1.0, 0.001, num_rounds).astype(np.float32)
scc = Cosine_SCC(k=5, num_rounds=num_rounds, thresholds=thresholds, index_name='cosine_sgtree', cores=cores, verbosity=0)
bs = 1
for i in range(0, x.shape[0], bs):
    # print(i)
    scc.partial_fit(x[i:i+bs])
b_t = gt() - t
print("Clustering time:", b_t, "seconds")
del scc
sys.stdout.flush()
