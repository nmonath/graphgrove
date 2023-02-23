<div align="center">
  <img src="https://raw.githubusercontent.com/nmonath/graphgrove/main/logo.png">
</div>

## Install

Linux wheels available (python >=3.6) on [pypi](https://pypi.org/project/graphgrove/):

```
pip install graphgrove
```

Building from source:

```
conda create -n gg python=3.8
conda activate gg
pip install numpy
make
```

To build your own wheel:

```
conda create -n gg python=3.8
conda activate gg
pip install numpy
make
pip install build
python -m build --wheel
# which can be used as:
# pip install --force dist/graphgrove-0.0.1-cp37-cp37m-linux_x86_64.whl 
```

## Examples

Toy examples of [clustering](examples/clustering.py), [DAG-structured clustering](examples/dag_clustering.py),  and [nearest neighbor search](examples/nearest_neighbor_search.py) are available. 

At a high level, incremental clustering can be done as:

```Python
import graphgrove as gg
k = 5
num_rounds = 50
thresholds = np.geomspace(1.0, 0.001, num_rounds).astype(np.float32)
scc = gg.vec_scc.Cosine_SCC(k=k, num_rounds=num_rounds, thresholds=thresholds, index_name='cosine_sgtree', cores=cores, verbosity=0)
# data_batches - generator of numpy matrices mini-batch-size by dim
for batch in data_batches:
    scc.partial_fit(batch)
```

Incremental nearest neighbor search can be done as:
```Python
import graphgrove as gg
k=5
cores=4
tree = gg.graph_builder.Cosine_SGTree(k=k, cores=cores)
# data_batches - generator of numpy matrices mini-batch-size by dim
for batch in data_batches:
    tree.insert(batch) # or tree.insert_and_knn(batch) 
```

## Algorithms Implemented

Clustering:
* Sub-Cluster Component Algorithm (SCC) and its minibatch variant from the paper: [Scalable Hierarchical Agglomerative Clustering](https://dl.acm.org/doi/10.1145/3447548.3467404). Nicholas, Monath, Kumar Avinava Dubey, Guru Guruganesh, Manzil Zaheer, Amr Ahmed, Andrew McCallum, Gokhan Mergen, Marc Najork Mert Terzihan Bryon Tjanaka Yuan Wang Yuchen Wu. KDD. 2021
* DAG Structured clustering (LLama) from [DAG-Structured Clustering by Nearest Neighbors](https://proceedings.mlr.press/v130/monath21a). Nicholas Monath, Manzil Zaheer, Kumar Avinava Dubey, Amr Ahmed, Andrew McCallum. AISTATS 2021.


Nearest Neighbor Search:
* CoverTree: Alina Beygelzimer, Sham Kakade, and John Langford. "Cover trees for nearest neighbor."  ICML. 2006.
* SGTree: SG-Tree is a new data structure for exact nearest neighbor search inspired from Cover Tree and its improvement, which has been used in the TerraPattern project. At a high level, SG-Tree tries to create a hierarchical tree where each node performs a "coarse" clustering. The centers of these "clusters" become the children and subsequent insertions are recursively performed on these children. When performing the NN query, we prune out solutions based on a subset of the dimensions that are being queried. This is particularly useful when trying to find the nearest neighbor in highly clustered subset of the data, e.g. when the data comes from a recursive mixture of Gaussians or more generally time marginalized coalscent process . The effect of these two optimizations is that our data structure is extremely simple, highly parallelizable and is comparable in performance to existing NN implementations on many data-sets. Manzil Zaheer, Guru Guruganesh, Golan Levin, Alexander Smola. [TerraPattern: A Nearest Neighbor Search Service](http://manzil.ml/res/Papers/2019_sgtree.pdf). 2019.
* DyNNIBAL / NysSG Tree: A variant of SG Tree which supports using Nystrom-based low-dimensional representations, dynamic rebuilding, and sampling. Nicholas Monath, Manzil Zaheer, Kelsey Allen, Andrew McCallum. "Improving Dual-Encoder Training through Dynamic Indexes for Negative Mining". AISTATS. 2023. 
