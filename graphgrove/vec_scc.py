"""
Copyright (c) 2021 The authors of SCC All rights reserved.

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
from graphgrove.scc import SCC
from graphgrove import graph_builder
import numpy as np


class Cosine_SCC(object):
  def __init__(self, k=25, num_rounds=50, thresholds=None, index_name='cosine_sgtree', cores=12, cc_alg=0, par_minimum=100000, verbosity=0, beam_size=100, hnsw_max_degree=200, hnsw_ef_search=200, hnsw_ef_construction=200):
    self.k = k
    self.num_rounds = num_rounds
    self.thresholds = thresholds
    self.index_name = index_name
    self.cores = cores
    if thresholds is None:
      self.thresholds = np.ones(num_rounds)*-np.Inf
    else:
     self.thresholds = thresholds
    self.thresholds = self.thresholds.astype(np.float32)
    self.cc_alg = cc_alg
    self.par_minimum = par_minimum
    self.verbosity = verbosity
    self.point_counter = 0
    self.total_scc_insert_time = 0

    self.beam_size = beam_size
    self.hnsw_max_degree = hnsw_max_degree
    self.hnsw_ef_search = hnsw_ef_search
    self.hnsw_ef_construction = hnsw_ef_construction
    
    if self.index_name.lower() == 'cosine_covertree':
      self.index = graph_builder.Cosine_CoverTree(self.k, self.cores)
    elif self.index_name.lower() == 'cosine_sgtree':
      self.index = graph_builder.Cosine_SGTree(self.k, self.cores)
    elif self.index_name.lower() == 'cosine_sgtreebeam':
     self.index = graph_builder.Cosine_SGTreeBeam(self.k, cores=self.cores, beam_size=self.beam_size)
    elif self.index_name.lower() == 'cosine_faissflat':
     self.index = graph_builder.Cosine_FaissFlat(self.k)
    elif self.index_name.lower() == 'cosine_faisshnsw':
      self.index = graph_builder.Cosine_FaissHNSW(self.k, self.hnsw_max_degree, self.hnsw_ef_search, self.hnsw_ef_construction)
    
    self.scc = SCC.init(self.thresholds, self.cores, self.cc_alg, self.par_minimum, self.verbosity)

#   def __del__(self):
#     del self.scc

  def partial_fit(self, vecs):
    self.index.insert_and_knn(vecs)
    g = self.index.latest_update
    if g is not None:
      t0 = time.time()
      self.scc.insert_graph_mb(g[0], g[1], g[2])
      t1 = time.time()
      self.total_scc_insert_time += t1-t0
    self.point_counter += vecs.shape[0]

  def add_edges(self, vecs):
    self.index.insert_and_knn(vecs)
    g = self.index.latest_update
    if g is not None:
      t0 = time.time()
      self.scc.add_edges(g[0], g[1], g[2])
      t1 = time.time()
      self.total_scc_insert_time += t1 - t0
    self.point_counter += vecs.shape[0]
    
  def update_on_edges(self):
    self.index.update_on_edges()