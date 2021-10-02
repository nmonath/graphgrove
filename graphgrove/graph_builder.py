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

from absl import logging
import numpy as np
from scipy.sparse import coo_matrix
import time
from graphgrove.covertree import NNS_L2 as CoverTree_NNS_L2
from graphgrove.sgtree import NNS_L2 as SGTree_NNS_L2

def to_coo(values, idx, offset, K):
    K = np.minimum(K, idx.shape[1])
    rows = np.reshape(np.repeat(np.expand_dims(np.arange(offset, offset+values.shape[0]), 1), K, axis=1), [-1])
    data = np.reshape(values, [-1])
    cols = np.reshape(idx, [-1])
    return data[rows!=cols], rows[rows!=cols], cols[rows!=cols]

def unit_norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms
    return X

class Index(object):
    def __init__(self, k):
        self.index = None
        self.row = None
        self.col = None
        self.data = None
        self.cached_topk_val = None
        self.cached_topk_idx = None
        self.k = k
        self.sim_graph = None
        self.latest_update = None
        self.num_points = 0
        self.total_knn_time = 0
        self.total_insert_time = 0
        self.total_graph_update_time = 0

    def topk(self, queries):
        pass

    def update_graph(self, new_vectors):
        id_of_vector_start = self.num_points - new_vectors.shape[0]
        t0 = time.time()
        values, indices = self.topk(new_vectors)
        t1 = time.time()
        self.total_knn_time += t1 - t0
        t0 = time.time()
        data, row, col = to_coo(values, indices, id_of_vector_start, self.k)
        if len(row) > 0:
            self.latest_update = [row,col, data]
        else:
            self.latest_update = None
        t1 = time.time()
        self.total_graph_update_time += t1-t0

class Cosine_CoverTree(Index):
    def __init__(self, k, cores=4, add_noise=True, noise_amount=1e-6, assume_unit_normed=True):
        super(Cosine_CoverTree, self).__init__(k)
        self.index = None
        self.row = None
        self.col = None
        self.data = None
        self.cached_topk_val = None
        self.cached_topk_idx = None
        self.k = k
        self.sim_graph = None
        self.cores = cores
        self.add_noise = add_noise
        self.two_vector_cache = None
        self.assume_unit_normed = assume_unit_normed
        self.noise_amount = noise_amount

    def build(self, vectors):
        c = 0
        if self.two_vector_cache is not None:
            c += self.two_vector_cache.shape[0]
        if vectors.shape[0] + c >= 2:
            t0 = time.time()
            if self.two_vector_cache is not None:
                vectors = np.concatenate([self.two_vector_cache, vectors])
                self.num_points -= self.two_vector_cache.shape[0]
                self.two_vector_cache = None
            if self.add_noise:
                vectors += np.random.randn(vectors.shape[0], vectors.shape[1]) * self.noise_amount
            self.num_points += vectors.shape[0]
            if not self.assume_unit_normed:
                vectors = unit_norm(vectors)
            self.index = CoverTree_NNS_L2.from_matrix(vectors, use_multi_core=self.cores)
            t1 = time.time()
            self.total_insert_time += t1 - t0
        else:
            self.two_vector_cache = vectors
            self.num_points += vectors.shape[0]

    def insert(self, vectors):
        if self.index is None:
            self.build(vectors)
        else:
            t0 = time.time()
            if self.add_noise:
                vectors += np.random.randn(vectors.shape[0], vectors.shape[1]) * self.noise_amount
            if not self.assume_unit_normed:
                vectors = unit_norm(vectors)
            self.index.insert(vectors, uid=np.arange(self.num_points, self.num_points+vectors.shape[0]), use_multi_core=self.cores)
            self.num_points += vectors.shape[0]
            t1 = time.time()
            self.total_insert_time += t1 - t0

    def insert_and_knn(self, vectors):
        if self.index is None:
            self.build(vectors)
        else:
            t0 = time.time()
            if self.add_noise:
                vectors += np.random.randn(vectors.shape[0], vectors.shape[1]) * self.noise_amount
            if not self.assume_unit_normed:
                vectors = unit_norm(vectors)
            self.index.insert(vectors, uid=np.arange(self.num_points, self.num_points+vectors.shape[0]), use_multi_core=self.cores)
            self.num_points += vectors.shape[0]
            t1 = time.time()
            self.total_insert_time += t1 - t0
        self.update_graph(vectors)

    def topk(self, query):
        if not self.assume_unit_normed:
            query = unit_norm(query)
        if self.index is None:
            scores = query @ self.two_vector_cache.T
            sorted_ord = np.argsort(scores, axis=1)
            topk_idx = sorted_ord[:, :np.minimum(self.k, self.two_vector_cache.shape[0])]
            return scores[topk_idx], topk_idx
        else:
            results = self.index.kNearestNeighbours(query, min(self.k, self.num_points), use_multi_core=self.cores)
            return (2 - results[1].astype(np.float32) ** 2) / 2, results[0].astype(np.int32)


class Cosine_SGTree(Cosine_CoverTree):
    def __init__(self, k, cores=4, add_noise=True, noise_amount=1e-6, assume_unit_normed=True):
        super(Cosine_SGTree, self).__init__(k, cores, add_noise, noise_amount, assume_unit_normed)

    def build(self, vectors):
        c = 0
        if self.two_vector_cache is not None:
            c += self.two_vector_cache.shape[0]
        if vectors.shape[0] + c >= 2:
            t0 = time.time()
            if self.two_vector_cache is not None:
                vectors = np.concatenate([self.two_vector_cache, vectors])
                self.num_points -= self.two_vector_cache.shape[0]
                self.two_vector_cache = None
            if self.add_noise:
                vectors += np.random.randn(vectors.shape[0], vectors.shape[1]) * self.noise_amount
            self.num_points += vectors.shape[0]
            if not self.assume_unit_normed:
                vectors = unit_norm(vectors)
            self.index = SGTree_NNS_L2.from_matrix(vectors, use_multi_core=self.cores)
            t1 = time.time()
            self.total_insert_time += t1 - t0
        else:
            self.two_vector_cache = vectors
            self.num_points += vectors.shape[0]

class Cosine_SGTreeBeam(Cosine_SGTree):
    def __init__(self, k, beam_size=100, cores=4, add_noise=True, noise_amount=1e-6, assume_unit_normed=True):
        super(Cosine_SGTreeBeam, self).__init__(k, cores, add_noise, noise_amount, assume_unit_normed)
        self.beam_size = beam_size

    def topk(self, query):
        if not self.assume_unit_normed:
            query = unit_norm(query)
        if self.index is None:
            scores = query @ self.two_vector_cache.T
            sorted_ord = np.argsort(scores, axis=1)
            topk_idx = sorted_ord[:, :np.minimum(self.k, self.two_vector_cache.shape[0])]
            return scores[topk_idx], topk_idx
        else:
            results = self.index.kNearestNeighboursBeam(query, min(self.k, self.num_points),
                                                        use_multi_core=self.cores, beam_size=self.beam_size)
            return (2-results[1].astype(np.float32)**2)/2, results[0].astype(np.int32)

class Cosine_FaissFlat(Index):
    def __init__(self, k, cores=4, add_noise=True, noise_amount=1e-6, assume_unit_normed=True):
        super(Cosine_FaissFlat, self).__init__(k)
        self.index = None
        self.row = None
        self.col = None
        self.data = None
        self.cached_topk_val = None
        self.cached_topk_idx = None
        self.k = k
        self.sim_graph = None
        self.cores = cores
        self.add_noise = add_noise
        self.noise_amount = noise_amount
        self.assume_unit_normed = assume_unit_normed

    def insert(self, vectors):
        if self.index is None:
            self.build(vectors)
        else:
            t0 = time.time()
            if self.add_noise:
                vectors += np.random.randn(vectors.shape[0], vectors.shape[1]) * self.noise_amount
            self.index.add(vectors)
            self.num_points += vectors.shape[0]
            t1 = time.time()
            self.total_insert_time += t1 - t0

    def build(self, vectors):
        t0 = time.time()
        if self.add_noise:
            vectors += np.random.randn(vectors.shape[0], vectors.shape[1]) * self.noise_amount
        self.num_points += vectors.shape[0]
        if not self.assume_unit_normed:
            vectors = unit_norm(vectors)
        import faiss
        self.index = faiss.IndexFlat(vectors.shape[1])
        self.index.verbose = True
        self.index.metric_type = faiss.METRIC_INNER_PRODUCT
        self.index.add(vectors)
        t1 = time.time()
        self.total_insert_time += t1 - t0

    def insert_and_knn(self, vectors):
        if self.index is None:
            self.build(vectors)
        else:
            t0 = time.time()
            if self.add_noise:
                vectors += np.random.randn(vectors.shape[0], vectors.shape[1]) * self.noise_amount
            if not self.assume_unit_normed:
                vectors = unit_norm(vectors)
            self.index.add(vectors)
            self.num_points += vectors.shape[0]
            t1 = time.time()
            self.total_insert_time += t1 - t0
        self.update_graph(vectors)

    def topk(self, query):
        results = self.index.search(query, min(self.k, self.num_points))
        return results[0].astype(np.float32), results[1].astype(np.int32)

class Cosine_FaissHNSW(Index):

    def __init__(self, k,
                 max_degree=128,
                 efSearch=128,
                 efConstruction=200,
                 add_noise=True,
                 noise_amount=1e-6,
                 assume_unit_norm=True):
        super(FaissHNSW, self).__init__(k)
        self.index = None
        self.row = None
        self.col = None
        self.data = None
        self.k = k
        self.sim_graph = None
        self.add_noise = add_noise
        self.max_degree = max_degree
        self.efSearch = efSearch
        self.efConstruction = efConstruction
        self.noise_amount = noise_amount
        self.assume_unit_norm = assume_unit_norm

    def build(self, vectors):
        t0 = time.time()
        if self.add_noise:
            vectors += np.random.randn(vectors.shape[0], vectors.shape[1]) * self.noise_amount
        self.num_points += vectors.shape[0]
        if not self.assume_unit_normed:
            vectors = unit_norm(vectors)
        self.index = faiss.IndexHNSWFlat(vectors.shape[1], self.max_degree)
        self.index.hnsw.efConstruction = self.efConstruction
        self.index.hnsw.efSearch = self.efSearch
        self.index.metric_type = faiss.METRIC_L2
        self.index.add(vectors)
        t1 = time.time()
        self.total_insert_time += t1 - t0

    def insert(self, vectors):
        if self.index is None:
            self.build(vectors)
        else:
            t0 = time.time()
            if self.add_noise:
                vectors += np.random.randn(vectors.shape[0], vectors.shape[1]) * self.noise_amount
            if not self.assume_unit_normed:
                vectors = unit_norm(vectors)
            self.index.add(vectors)
            self.num_points += vectors.shape[0]
            t1 = time.time()
            self.total_insert_time += t1 - t0

    def insert_and_knn(self, vectors):
        if self.index is None:
            self.build(vectors)
        else:
            t0 = time.time()
            if self.add_noise:
                vectors += np.random.randn(vectors.shape[0], vectors.shape[1]) * self.noise_amount
            if not self.assume_unit_normed:
                vectors = unit_norm(vectors)
            self.index.add(vectors)
            self.num_points += vectors.shape[0]
            t1 = time.time()
            self.total_insert_time += t1 - t0
        self.update_graph(vectors)

    def topk(self, query):
        if not self.assume_unit_normed:
            query = unit_norm(query)
        results = self.index.search(query, min(self.k, self.num_points))
        return (2-results[0].astype(np.float32)**2)/2, results[1].astype(np.int32)

