"""
Copyright (c) 2021 The authors of SCC All rights reserved.

Initially modified from cover_tree.py of CoverTree
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

import sccc

import numpy as np

class Node(object):
  import sccc
  """SCC node from c++."""
  base_vars = ['this', 'uid', 'height', 'mean']

  def __init__(self, this):
    info = sccc.node_property(this)
    info['this'] = this
    if info['others']:
      others = pickle.loads(info['others'])
      info.update(others)
    del info['others']
    self.__dict__ = info

  @property
  def children(self):
    return [Node(child) for child in sccc.node_children(self.this)]

  @property
  def pruned_children(self):
    return [Node(child) for child in sccc.pruned_children(self.this)]

  def descendants(self):
    return sccc.descendants(self.this)

  def __setattr__(self, name, value):
    if name not in self.base_vars:
      super(Node, self).__setattr__(name, value)
      props = {k: v for k, v in vars(self).items() if k not in self.base_vars}
      if props:
        sccc.node_save(self.this, pickle.dumps(props))
    else:
      print('Cannot set {}'.format(name))


class Level(object):
  import sccc
  """SCC node from c++."""
  base_vars = ['this', 'height']

  def __init__(self, this):
    info = sccc.level_property(this)
    info['this'] = this
    self.__dict__ = info

  @property
  def nodes(self):
    return [Node(n) for n in sccc.level_nodes(self.this)]

class SCC(object):
  import sccc
  
  def __init__(self, this):
    self.this = this

  def __del__(self):
    sccc.delete(self.this)

  @property
  def levels(self):
    return [Level(l) for l in sccc.levels(self.this)]

  def fit(self):
    sccc.fit(self.this)

  def set_marking_strategy(self, strat):
    sccc.set_marking_strategy(self.this, strat)

  def knn_time(self):
    return sccc.knn_time(self.this)

  def update_time(self):
    return sccc.update_time(self.this)

  def graph_update_time(self):
    return sccc.graph_update_time(self.this)

  def overall_update_time(self):
    return sccc.overall_update_time(self.this)

  def best_neighbor_time(self):
    return sccc.best_neighbor_time(self.this)

  def cc_time(self):
    return sccc.cc_time(self.this)

  def center_update_time(self):
    return sccc.center_update_time(self.this)

  def number_marked(self):
    return sccc.number_marked(self.this)

  def max_number_marked(self):
    return sccc.max_number_marked(self.this)

  def max_number_cc_iterations(self):
    return sccc.max_number_cc_iterations(self.this)

  def sum_number_cc_iterations(self):
    return sccc.sum_number_cc_iterations(self.this)

  def sum_number_cc_edges(self):
    return sccc.sum_number_cc_edges(self.this)    

  def sum_number_cc_nodes(self):
    return sccc.sum_number_cc_nodes(self.this)

  def total_number_nodes(self):
    return sccc.total_number_nodes(self.this)

  def insert(self, matrix, uids, cores=4, k=25, beam=50):
    sccc.insert(self.this, matrix, uids, k, cores, beam)

  def insert_mb(self, matrix, uids, cores=4, k=25, beam=50):
    sccc.insert_mb(self.this, matrix, uids, k, cores, beam)

  def add_edges(self, row, col, sim):
    if len(row.shape) == 1:
      row = row[:, None]
    if len(col.shape) == 1:
      col = col[:, None]
    if len(sim.shape) == 1:
      sim = sim[:, None]
    sccc.add_graph_edges_mb(self.this, row.astype(np.uint32), col.astype(np.uint32), sim.astype(np.float32))

  def update_on_edges(self):
    sccc.update(self.this)

  def fit_on_large_batch(self, n, row, col, sim):
    if len(row.shape) == 1:
      row = row[:, None]
    if len(col.shape) == 1:
      col = col[:, None]
    if len(sim.shape) == 1:
      sim = sim[:, None]
    sccc.fit_on_large_batch(self.this, n, row.astype(np.uint32), col.astype(np.uint32), sim.astype(np.float32))

  def insert_graph_mb(self, row, col, sim):
    if len(row.shape) == 1:
      row = row[:, None]
    if len(col.shape) == 1:
      col = col[:, None]
    if len(sim.shape) == 1:
      sim = sim[:, None]
    sccc.insert_graph_mb(self.this, row.astype(np.uint32), col.astype(np.uint32), sim.astype(np.float32))

  def roots(self):
    return [Node(x) for x in sccc.roots(self.this)]

  @classmethod
  def init(cls, thresholds, cores=1, cc_alg=1, pararallel_min_size=50000, verbosity=0):
    ptr = sccc.init(thresholds, cores, cc_alg, pararallel_min_size, verbosity)
    return cls(ptr)