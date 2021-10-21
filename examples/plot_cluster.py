import time
import sys

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from graphgrove.vec_scc import Cosine_SCC
from graphgrove.graph_builder import unit_norm
from fairseq.data import Dictionary
dictionary = Dictionary.load('../knnlm-distill/data-bin/wikitext103-bpe/dict.txt')

to_save = ['bank', 'shore', 'institution', 'beautiful']

gt = time.time

np.random.seed(123)
cores = 80

for word in to_save:
    plot_x = []
    plot_y = []

    idx = dictionary.index(word)
    print(idx)
    vectors = np.load('../knnlm-distill/dstore/ids/' + str(idx) + '.npy')
    print(vectors.shape)
    vectors = unit_norm(vectors)
    vectors = vectors.astype(np.float32)
    t = gt()
    num_rounds = 50
    thresholds = np.geomspace(1.0, 0.001, num_rounds).astype(np.float32)
    scc = Cosine_SCC(k=25, num_rounds=num_rounds, thresholds=thresholds,
                     index_name='cosine_faisshnsw', cores=cores, verbosity=0)
    scc.partial_fit(vectors)
    b_t = gt() - t
    scc = scc.scc  # cos_scc operates on vectors, it's member object, scc (https://github.com/nmonath/graphgrove/blob/main/graphgrove/scc.py) operates on the k-nearest neighbor graph.
    levels = scc.levels  # clustering will store the flat clustering
    for selected_level in range(num_rounds + 1):
        clustering = []
        for node in levels[selected_level].nodes:
            clustering.append(
                node.descendants())  # descendants has the ids of the data points which are the descendant leaves
        number_clusters = len(clustering)
        print(selected_level, number_clusters)
        plot_x.append(selected_level)
        plot_y.append(number_clusters)

    plt.scatter(plot_x, plot_y, label=word, s=4)
plt.legend(loc="upper right")
plt.ylim([0, 1000])
plt.savefig('words.pdf')
