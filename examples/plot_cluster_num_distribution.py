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
print(len(thresholds))
plot_x = []
plot_y = []

plot_detail_x = []
plot_detail_y = []

for idx in tqdm.tqdm(range(50, 10000, 50)):
    cluster_result = torch.load('../knnlm-distill/dstore/clusters/' + str(idx) + '.pt')
    for level, cluster in enumerate(cluster_result['cluster_levels']):
        # if level > 0:
        #     plot_x.append(thresholds[level-1])
        #     plot_y.append(len(cluster))
        if level > 40:
            plot_detail_x.append(thresholds[level-1])
            plot_detail_y.append(len(cluster))

# plt.scatter(plot_x, plot_y, s=3)
# plt.savefig('full_dist.pdf')
#
# plt.clf()


plt.scatter(plot_detail_x, plot_detail_y, s=3)
plt.savefig('right_dist_smpl.pdf')

plt.clf()

