import time
import sys

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch

from graphgrove.vec_scc import Cosine_SCC
from graphgrove.graph_builder import unit_norm

gt = time.time

np.random.seed(123)
cores = 80

num_rounds = 50
thresholds = np.geomspace(1.0, 0.001, num_rounds).astype(np.float32)

for idx in tqdm.tqdm(range(50, 10000)):
    vectors = np.load('../knnlm-distill/dstore/ids/' + str(idx) + '.npy')
    print(vectors.shape)
    vectors = unit_norm(vectors)
    vectors = vectors.astype(np.float32)

    t = gt()
    scc = Cosine_SCC(k=25, num_rounds=num_rounds, thresholds=thresholds,
                     index_name='cosine_faisshnsw', cores=cores, verbosity=0)
    scc.partial_fit(vectors)
    b_t = gt() - t
    scc = scc.scc
    # cos_scc operates on vectors, it's member object, scc
    # (https://github.com/nmonath/graphgrove/blob/main/graphgrove/scc.py) operates on the k-nearest neighbor graph.
    levels = scc.levels  # clustering will store the flat clustering

    cluster_data_save = {'thresholds': thresholds,
                         'cluster_levels': []}

    for selected_level in range(num_rounds + 1):
        clustering = []
        for node in levels[selected_level].nodes:
            clustering.append(
                node.descendants().squeeze(-1))  # descendants has the ids of the data points which are the descendant leaves
        cluster_data_save['cluster_levels'].append(clustering)

    torch.save(cluster_data_save, '../knnlm-distill/dstore/clusters/' + str(idx) + '.pt')