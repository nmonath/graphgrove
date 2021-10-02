"""
Copyright (c) 2021 The authors of Llama All rights reserved.

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

import llamac
import numpy as np

class LLAMA(object):
  def __init__(self, this):
    self.this = this

  def __del__(self):
    llamac.delete(self.this)

  def cluster(self): 
    """Run the DAG-clustering process."""
    llamac.cluster(self.this)

  def assignments(self):
    """Return clusters of the DAG-structure discovered.
    
    Returns:
    coo_matrix of size N by K where N is 
    the number of data points and K is the number
    of nodes in the DAG. coo_matrix[i,j] = 1 if 
    the point i is a descendant of the node j (i.e.,
    the cluster represented by node j contains point i).
    """
    return llamac.get_descendants(self.this)

  def structure(self):
    """Return edges of the DAG-structure discovered.
    
    Returns:
    a numpy matrix of size M by 2 where M is the number of
    edges. each row of the has a value of [child_node_id, parent_node_id]
    """
    return llamac.get_child_parent_edges(self.this)

  def round(self, r):
    """Return the cover of the r^{th} round.
    
    Returns:
    coo_matrix of size N by K_r where N is 
    the number of data points and K_r is the number
    of nodes in r^{th} round. coo_matrix[i,j] = 1 if 
    the point i is a descendant of the node j (i.e.,
    the cluster represented by node j contains point i).
    """
    return llamac.get_round(self.this, r)

  @classmethod
  def from_graph(cls, coo_graph, 
           num_rounds, cores=4, linkage=2, 
           max_num_parents=5, max_num_neighbors=100, 
           thresholds=None, lowest_value=-10000):
    """Instantiate a LLAMA object with the given graph & hyperparameters.

    Arguments:
    coo_graph -- the graph to cluster.
    num_rounds -- the number of rounds to use.

    Keyword arguments:
    cores -- number of parallel threads to use (default 4).
    linkage -- linkage function to use either integer (0 for single, 1 for average, 2 for approx. average). (default 2).
          or string valued ('single', 'average, 'approx_average')
    max_num_parents -- maximum number of parents any node can have (default 5).
    max_num_neighbors -- maximum number of neigbhors any node can have in the graph (default 100).
    thresholds -- None (for no threshold use). Or a numpy array (float32) of the minimum similarity to allow in an agglomeration (default None).
    lowest_value -- value used for missing / minimum similarity (default -10000)
    """
    rows, cols, sims = coo_graph.row.astype(np.uint32), coo_graph.col.astype(np.uint32), coo_graph.data.astype(np.float32)
    if len(rows.shape) == 1:
      rows = rows[:, None]
    if len(cols.shape) == 1:
      cols = cols[:, None]
    if len(sims.shape) == 1:
      sims = sims[:, None]
    
    if thresholds is None:
      thresholds = np.ones(num_rounds, dtype=np.float32) * lowest_value
    if type(linkage) == str:
      if linkage.lower() == 'single':
        linkage = 0
      elif linkage.lower() == 'average':
        linkage = 1
      elif linkage.lower() == 'approx_average':
        linkage = 2
      else:
        raise Exception('Unknown linkage %s. Options are single, average, approx_average' % linkage)
    ptr = llamac.new(rows, cols, sims, linkage, num_rounds, thresholds, cores, max_num_parents, max_num_neighbors, lowest_value)
    return cls(ptr)
