import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from multiprocessing import Pool
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

levels_interested = [40, 45, 50]
histo_data = dict.fromkeys(levels_interested)
for x in histo_data:
    histo_data[x] = []

# def process_cluster_id(cluster_id):
#     cluster_result = torch.load('../knnlm-distill/dstore/clusters/' + str(cluster_id) + '.pt')
#     for level, cluster in enumerate(cluster_result['cluster_levels']):
#         if level in levels_interested:
#             histo_data[level].append(len(cluster))
#
# with Pool(processes=60) as pool:
bins = np.linspace(0, 40, 20)
print(bins)

for idx in tqdm.tqdm(range(50, 10000, 10)):
    cluster_result = torch.load('../knnlm-distill/dstore/clusters/' + str(idx) + '.pt')
    for level, cluster in enumerate(cluster_result['cluster_levels']):
        if level in levels_interested:
            num_cls = len(cluster)
            if num_cls > 40:
                num_cls = 40
            histo_data[level].append(num_cls)

for l in histo_data:
    print(np.sum(histo_data[l])*10)
    plt.hist(histo_data[l], bins, alpha=0.5, label=thresholds[l-1])
plt.legend(loc='upper right')
# plt.ylim(0, 1000)
plt.savefig('histogram_num_clusters.pdf')

plt.clf()

