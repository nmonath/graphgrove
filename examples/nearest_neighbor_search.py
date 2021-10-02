"""
Copyright (c) 2021 The authors of SG Tree All rights reserved.

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
import numpy as np

from graphgrove.sgtree import NNS_L2 as SGTree_NNS_L2
from graphgrove.covertree import NNS_L2 as CoverTree_NNS_L2

gt = time.time

np.random.seed(123)
cores = 4

print('======== Building Dataset ==========')
N=1000
K=10
D=784
means = 20*np.random.rand(K,D) - 10
x = np.vstack([np.random.randn(N,D) + means[i] for i in range(K)])
np.random.shuffle(x)
x = x.astype(np.float32)
x = np.require(x, requirements=['A', 'C', 'O', 'W'])
y = np.vstack([np.random.randn(N,D) + means[i] for i in range(K)])
y = y.astype(np.float32)
y = np.require(y, requirements=['A', 'C', 'O', 'W'])

print('======== Cover Tree ==========')
t = gt()
ct = CoverTree_NNS_L2.from_matrix(x, use_multi_core=cores)
b_t = gt() - t
#ct.display()
print("Building time:", b_t, "seconds")

print('Test k-Nearest Neighbours - Exact (k=3): ')
t = gt()
idx1, d1 = ct.kNearestNeighbours(y,3, use_multi_core=cores)
b_t = gt() - t
print("Query time - Exact:", b_t, "seconds")

print('======== SG Tree ==========')
t = gt()
ct = SGTree_NNS_L2.from_matrix(x, use_multi_core=cores)
b_t = gt() - t
#ct.display()
print("Building time:", b_t, "seconds")

print('Test k-Nearest Neighbours - Exact (k=3): ')
t = gt()
idx1, d1 = ct.kNearestNeighbours(y,3, use_multi_core=cores)
b_t = gt() - t
print("Query time - Exact:", b_t, "seconds")

print('Test k-Nearest Neighbours - Beam (k=3, beam_size=10): ')
t = gt()
idx1, d1 = ct.kNearestNeighboursBeam(y, 3, use_multi_core=cores, beam_size=10)
b_t = gt() - t
print("Query time - Beam:", b_t, "seconds")

print('Test Range - cores=0')
t = gt()
idx1, d1 = ct.RangeSearch(y, r=0.5, use_multi_core=0)
b_t = gt() - t
print("Query time - Range:", b_t, "seconds")

print('Test Range - cores=%s' % cores)
t = gt()
idx1, d1 = ct.RangeSearch(y, r=0.5, use_multi_core=cores)
b_t = gt() - t
print("Query time - Range:", b_t, "seconds")