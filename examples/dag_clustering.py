"""
Copyright (c) 2021 The authors of LLama All rights reserved.

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

from graphgrove.llama import LLAMA
import numpy as np
from scipy.sparse import coo_matrix

num_nodes = 100
num_edges = 2500
r = np.random.choice(num_nodes, size=num_edges)
c = np.random.choice(num_nodes, size=num_edges)
sim = np.random.random_sample(size=num_edges).astype(np.float32)
graph = coo_matrix((sim,(r,c)))

def make_symmetric(coo_mat):
    lil = coo_mat.tolil()
    rows, cols = lil.nonzero()
    lil[cols, rows] = lil[rows, cols].maximum(lil[cols, rows])
    return lil.tocoo()

graph = make_symmetric(graph)

llama = LLAMA.from_graph(graph, num_rounds=10, cores=3, linkage='approx_average')
llama.cluster()

print(llama.assignments())
print(llama.structure())
print(llama.round(2))