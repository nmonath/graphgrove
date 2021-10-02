"""
Copyright (c) 2021 The authors of SG Tree All rights reserved.

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

import covertreec


class Node(object):
  """CoverTree node from c++."""
  base_vars = ['this', 'uid', 'level', 'point', 'maxdistUB']

  def __init__(self, this):
    info = covertreec.node_property(this)
    info['this'] = this
    if info['others']:
      others = pickle.loads(info['others'])
      info.update(others)
    del info['others']
    self.__dict__ = info

  @property
  def children(self):
    return [Node(child) for child in covertreec.node_children(self.this)]

  def __setattr__(self, name, value):
    if name not in self.base_vars:
      super(Node, self).__setattr__(name, value)
      props = {k: v for k, v in vars(self).items() if k not in self.base_vars}
      if props:
        covertreec.node_save(self.this, pickle.dumps(props))
    else:
      print('Cannot set {}'.format(name))

class NNS_L2(object):
  """CoverTree Class for NN search in Euclidean distance."""

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
    covertreec.delete(self.this)

  def __reduce__(self):
    buff = self.serialize()
    return (NNS_L2.from_string, (buff,))
    
  def __len__(self):
    return covertreec.size(self.this)

  @classmethod
  def from_matrix(cls, points, trunc=-1, use_multi_core=-1):
    ptr = covertreec.new(points, trunc, use_multi_core)
    return cls(ptr)

  @classmethod
  def from_string(cls, buff):
    ptr = covertreec.deserialize(buff)
    return cls(ptr)

  def insert(self, point, uid=None, use_multi_core=-1):
    if len(point.shape) == 1:
      return covertreec.insert(self.this, point, -1 if uid is None else uid)
    elif len(point.shape) == 2:
      if uid is None:
        N = covertreec.size(self.this)
        uid = np.arange(N, N + point.shape[0])
      return covertreec.batchinsert(self.this, point, uid, use_multi_core)
    else:
      print("Points to be inserted should be 1D or 2D matrix!")

  def remove(self, point):
    return covertreec.remove(self.this, point)

  def NearestNeighbour(self, points, use_multi_core=-1, return_points=False):
    return covertreec.NearestNeighbour(self.this, points, use_multi_core,
                                       return_points)

  def kNearestNeighbours(self,
                         points,
                         k=10,
                         use_multi_core=-1,
                         return_points=False):
    return covertreec.kNearestNeighbours(self.this, points, k, use_multi_core,
                                         return_points)

  def RangeSearch(self,
                  points,
                  r=1.0,
                  use_multi_core=-1,
                  return_points=False):
    return covertreec.RangeSearch(self.this, points, r, use_multi_core,
                                  return_points)

  def serialize(self):
    return covertreec.serialize(self.this)

  def display(self):
    return covertreec.display(self.this)

  def stats(self):
    return covertreec.stats(self.this)

  def test_covering(self):
    return covertreec.test_covering(self.this)
  
  def test_nesting(self):
    return covertreec.test_nesting(self.this)

  def spreadout(self, k):
    return covertreec.spreadout(self.this, k)
  
  def get_root(self):
    return Node(covertreec.get_root(self.this))

class MIPS(NNS_L2):
  """CoverTree Class for maximum inner product search."""

  def __init__(self, this, phi2):
    super(MIPS, self).__init__(this)
    self.phi2 = phi2

  def __reduce__(self):
    buff = self.serialize()
    return (MIPS.from_string, (buff, self.phi2))
    
  def __len__(self):
    return covertreec.size(self.this)

  @classmethod
  def from_matrix(cls, points, trunc=-1, user_max=None, use_multi_core=-1):
    # Find norm of points
    norm2 = (points**2).sum(1)
    phi2 = np.max(norm2) if user_max is None else user_max
    modified_points = np.hstack((points, np.sqrt(phi2 - norm2)[:, np.newaxis]))
    ptr = covertreec.new(modified_points, trunc, use_multi_core)
    return cls(ptr, phi2)

  @classmethod
  def from_string(cls, buff, phi2):
    ptr = covertreec.deserialize(buff)
    return cls(ptr, phi2)

  def insert(self, point, uid=None, use_multi_core=-1):
    if len(point.shape) == 1:
      norm2 = np.dot(point, point)
      modified_point = np.append(point, np.sqrt(self.phi2 - norm2))
      return covertreec.insert(self.this, modified_point, -1 if uid is None else uid)
    elif len(point.shape) == 2:
      if uid is None:
        N = covertreec.size(self.this)
        uid = np.arange(N, N + point.shape[0])
      norm2 = (point**2).sum(1)
      modified_points = np.hstack((point, np.sqrt(self.phi2 - norm2)[:, np.newaxis]))
      return covertreec.batchinsert(self.this, modified_points, uid, use_multi_core)
    else:
      print("Points to be inserted should be 1D or 2D matrix!")

  def remove(self, point):
    norm2 = np.dot(point, point)
    modified_point = np.append(point, np.sqrt(self.phi2 - norm2))
    return covertreec.remove(self.this, modified_point)

  def NearestNeighbour(self, points, use_multi_core=-1, return_points=False):
    modified_points = np.hstack(
        (points, np.zeros((points.shape[0], 1), dtype=points.dtype)))
    ret_val = list(
        covertreec.NearestNeighbour(self.this, modified_points, use_multi_core,
                                    return_points))
    norm2 = (points**2).sum(1)
    ret_val[1] = 0.5 * (self.phi2 + norm2 - ret_val[1]**2)
    return tuple(ret_val)

  def kNearestNeighbours(self,
                         points,
                         k=10,
                         use_multi_core=-1,
                         return_points=False):
    """main nearest neighbor function."""
    modified_points = np.hstack(
        (points, np.zeros((points.shape[0], 1), dtype=points.dtype)))
    import time 
    st_t = time.time()
    ret_val = list(
        covertreec.kNearestNeighbours(self.this, modified_points, k,
                                      use_multi_core, return_points))
    st_e = time.time()
    norm2 = (points**2).sum(1)[:, np.newaxis]
    ret_val[1] = 0.5 * (self.phi2 + norm2 - ret_val[1]**2)
    return tuple(ret_val)

  def RangeSearch(self,
                  points,
                  r=1.0,
                  use_multi_core=-1,
                  return_points=False):
    raise NotImplementedError('Range for MIPS not clear')

class MCSS(NNS_L2):
  """CoverTree Class for NN search in cosine distance."""

  def __init__(self, this):
    super(MCSS, self).__init__(this)

  def __reduce__(self):
    buff = self.serialize()
    return (MCSS.from_string, (buff,))

  @classmethod
  def from_matrix(cls, points, trunc=-1, use_multi_core=-1):
    # Find norm of points
    norm = np.sqrt((points**2).sum(1))
    modified_points = points / norm[:, np.newaxis]
    ptr = covertreec.new(modified_points, trunc, use_multi_core)
    return cls(ptr)

  @classmethod
  def from_string(cls, buff):
    ptr = covertreec.deserialize(buff)
    return cls(ptr)

  def insert(self, point):
    norm = np.sqrt(np.dot(point, point))
    modified_point = point / norm
    return covertreec.insert(self.this, modified_point)

  def remove(self, point):
    norm = np.sqrt(np.dot(point, point))
    modified_point = point / norm
    return covertreec.remove(self.this, modified_point)

  def NearestNeighbour(self, points, use_multi_core=-1, return_points=False):
    ret_val = list(
        covertreec.NearestNeighbour(self.this, points, use_multi_core,
                                    return_points))
    norm = np.sqrt((points**2).sum(1))
    ret_val[1] = 0.5 * (1.0 / norm + norm - ret_val[1]**2 / norm)
    return tuple(ret_val)

  def kNearestNeighbours(self,
                         points,
                         k=10,
                         use_multi_core=-1,
                         return_points=False):
    """main nearest neighbor function."""
    ret_val = list(
        covertreec.kNearestNeighbours(self.this, points, k, use_multi_core,
                                      return_points))
    norm = np.sqrt((points**2).sum(1))[:, np.newaxis]
    ret_val[1] = 0.5 * (1.0 / norm + norm - ret_val[1]**2 / norm)
    return tuple(ret_val)

  def RangeSearch(self,
                  points,
                  r=1.0,
                  use_multi_core=-1,
                  return_points=False):
    raise NotImplementedError('Range for MIPS not clear')
