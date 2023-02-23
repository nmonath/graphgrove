"""
Copyright (c) 2021 The authors of SG Tree All rights reserved.
Copyright (c) 2023 The authors of NysSG Tree All rights reserved.

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

import pickle
import numpy as np

import nyssgtreec as sgtreec


class Node(object):
  """SGTree node from c++."""
  base_vars = ['this', 'uid', 'level', 'point', 'maxdistUB']

  def __init__(self, this):
    info = sgtreec.node_property(this)
    info['this'] = this
    if info['others']:
      others = pickle.loads(info['others'])
      info.update(others)
    del info['others']
    self.__dict__ = info

  @property
  def children(self):
    return [Node(child) for child in sgtreec.node_children(self.this)]

  def __setattr__(self, name, value):
    if name not in self.base_vars:
      super(Node, self).__setattr__(name, value)
      props = {k: v for k, v in vars(self).items() if k not in self.base_vars}
      if props:
        sgtreec.node_save(self.this, pickle.dumps(props))
    else:
      print('Cannot set {}'.format(name))

class NNS_Nys(object):
  """NysSGTree Class for NN search with Nystrom Embeddings."""

  def __init__(self, this):
    if isinstance(this, tuple):
      self.this = this[0]
      self.root = Node(this[1])
    elif isinstance(this, int):
      self.this = this
      self.root = None
    else:
      raise NotImplementedError('this pointer should be int or tuple')

  def __del__(self):
    sgtreec.delete(self.this)

  def __reduce__(self):
    buff = self.serialize()
    return (NNS_L2.from_string, (buff,))
    
  def __len__(self):
    return sgtreec.size(self.this)

  @classmethod
  def from_matrix(cls, points, pointsProj, trunc=-1, use_multi_core=-1):
    ptr = sgtreec.new(points, pointsProj, trunc, use_multi_core)
    return cls(ptr)

  @classmethod
  def from_string(cls, buff):
    ptr = sgtreec.deserialize(buff)
    return cls(ptr)

  def insert(self, point, uid=None, use_multi_core=-1):
    if len(point.shape) == 1:
      return sgtreec.insert(self.this, point, -1 if uid is None else uid)
    elif len(point.shape) == 2:
      if uid is None:
        N = sgtreec.size(self.this)
        uid = np.arange(N, N + point.shape[0])
      return sgtreec.batchinsert(self.this, point, uid, use_multi_core)
    else:
      print("Points to be inserted should be 1D or 2D matrix!")

  def remove(self, point):
    return sgtreec.remove(self.this, point)

  def NearestNeighbour(self, points, use_multi_core=-1, return_points=False):
    return sgtreec.NearestNeighbour(self.this, points, use_multi_core,
                                       return_points)

  def kNearestNeighbours(self,
                         points,
                         k=10,
                         use_multi_core=-1,
                         return_points=False):
    return sgtreec.kNearestNeighbours(self.this, points, k, use_multi_core,
                                         return_points)

  def kNearestNeighboursBeam(self,
                         points,
                         k=10,
                         beam_size=100,
                         use_multi_core=-1,
                         return_points=False):
    return sgtreec.kNearestNeighboursBeam(self.this, points, k, use_multi_core,
                                         return_points, beam_size)

  def RangeSearch(self,
                  points,
                  r=1.0,
                  use_multi_core=-1,
                  return_points=False):
    return sgtreec.RangeSearch(self.this, points, r, use_multi_core,
                                  return_points)

  def serialize(self):
    return sgtreec.serialize(self.this)

  def display(self):
    return sgtreec.display(self.this)

  def stats(self):
    return sgtreec.stats(self.this)

  def test_covering(self):
    return sgtreec.test_covering(self.this)

  def test_nesting(self):
    return sgtreec.test_nesting(self.this)

  def spreadout(self, k):
    return sgtreec.spreadout(self.this, k)
  
  def get_root(self):
    return Node(sgtreec.get_root(self.this))
  
  def set_num_descendants(self):
    return sgtreec.set_num_descendants(self.this)

  def rejectionSample(self, points, num_samples=10, use_multi_core=-1,
                      return_points=False):
    return sgtreec.rejectionSample(self.this, points, num_samples,
                                   use_multi_core, return_points)

  def mhClusterSample(self, points,
                            k=10,
                            until_level=-10,
                            beamSize=100,
                            num_chains=10,
                            chain_length=10,
                            use_multi_core=-1,
                            return_points=False):
    return sgtreec.mhClusterSample(self.this, points, k,
                                   use_multi_core, return_points, 
                                   until_level, beamSize, num_chains, 
                                   chain_length)

  def mhClusterSampleHeuristic1(self,
                               points,
                               num_samples=10,
                               until_level=-10,
                               beamSize=100,
                               num_chains=10,
                               use_multi_core=-1,
                               return_points=False):
    return sgtreec.mhClusterSampleHeuristic1(self.this, points, num_samples,
                                   use_multi_core, return_points, 
                                   until_level, beamSize, num_chains)
    
  def mhClusterSampleHeuristic2(self,
                               points,
                               num_samples=10,
                               until_level=-10,
                               beamSize=100,
                               num_chains=10,
                               repeats=10,
                               use_multi_core=-1,
                               return_points=False):
    return sgtreec.mhClusterSampleHeuristic2(self.this, points, num_samples,
                                   use_multi_core, return_points,
                                   until_level, beamSize, num_chains, repeats)

  def update(self, points, points_proj, level):
    self.update_vector(points, points_proj)
    self.rebuild_level(level)

  def rebuild_level(self, level):
    sgtreec.rebuildLevel(self.this, level)

  def update_vector(self, points, points_proj):
    return sgtreec.updateVectors(self.this, points, points_proj)
  
