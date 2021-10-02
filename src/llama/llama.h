/*
 * Copyright (c) 2021 The authors of Llama All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _LLAMA_H
#define _LLAMA_H

//#define DEBUG

#include <atomic>
#include <fstream>
#include <iostream>
#include <stack>
#include <map>
#include <unordered_map>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>
#include <shared_mutex>
#include <string>
#include <chrono>
#include <queue>
#include <bitset>
#include <chrono>
#include <queue>

#ifdef __clang__
#define SHARED_MUTEX_TYPE shared_mutex
#else
#define SHARED_MUTEX_TYPE shared_timed_mutex
#endif

#include <Eigen/Core>

#include "utils.h"

#ifdef _FLOAT64_VER_
#define MY_NPY_FLOAT NPY_FLOAT64
typedef Eigen::MatrixXd matrixType;
typedef Eigen::VectorXd pointType;
typedef Eigen::VectorXd::Scalar scalar;
#else
#define MY_NPY_FLOAT NPY_FLOAT32
typedef Eigen::MatrixXf matrixType;
typedef Eigen::VectorXf pointType;
typedef Eigen::VectorXf::Scalar scalar;
#endif

typedef uint32_t point_id_t;
typedef uint32_t node_id_t;

class LLAMA
{
public:
    static LLAMA *from_graph(
        std::vector<uint32_t> r,
        std::vector<uint32_t> c,
        std::vector<Eigen::VectorXf::Scalar> s,
        unsigned linkage,
        unsigned num_rounds,
        scalar *thresholds,
        unsigned cores,
        unsigned max_num_parents,
        unsigned max_num_neighbors,
        scalar lowest_value);

    LLAMA(
        std::vector<uint32_t> r,
        std::vector<uint32_t> c,
        std::vector<Eigen::VectorXf::Scalar> s,
        unsigned linkage,
        unsigned num_rounds,
        scalar *thresholds,
        unsigned cores,
        unsigned max_num_parents,
        unsigned max_num_neighbors,
        scalar lowest_value);

    // the maximum total allowable number of nodes
    const static size_t MAX_NODES = 2000000;
    scalar lowest_value = -100000.0;

    // 0=single, 1=set average, 2=complete, 3=bag average
    unsigned linkage = 1;

    std::shared_timed_mutex mtx;

    // the maximum number of parents in the current round.
    unsigned max_num_parents;

    // the maximum number of neighbors that any noe should have
    unsigned max_num_neighbors;

    // the round of the algorithm that we are currently on.
    unsigned round_id = 0;

    // the total number of points in the dataset
    unsigned num_points;

    // the number of threads to use in parallelism
    unsigned cores;

    // the number of rounds to run in clustering
    unsigned num_rounds;

    // the thresholds that we use in clustering in each round (Llama as written in paper though does not use.)
    scalar *thresholds;

    bool clustering_run = false;

    class LLAMANode
    {
    public:

        // neighbors
        // node id and unnormalized count.
        std::map<LLAMANode *, scalar> neighbors;
        std::map<LLAMANode *, scalar> new_neighbors;


        // used only by avg_set linkage
        std::set<node_id_t> descendants;
        std::set<node_id_t> new_descendants;
        std::map<node_id_t, scalar> leaf_sims;
        std::set<node_id_t> leafs;
        // CC neighbor edges
        std::unordered_map<node_id_t, scalar> cc_edges;

        // parents
        std::vector<std::pair<LLAMANode *, scalar>> parents;

        // id of this node
        node_id_t ID;

        // one NN edge
        LLAMANode * best_neighbor;
        scalar best_neighbor_score;

        scalar count;
        LLAMANode * chosen_parent = NULL;
        scalar parent_count;
        scalar new_node_count;

        bool skip = false;

        std::shared_timed_mutex cclock;

        LLAMANode(node_id_t uid) {
            ID = uid;
        }
    };

    void cluster();
    void perform_round(scalar threshold);
    void propose_parents();
    void one_nn(scalar threshold);
    void contract();
    void contract_bag_average();
    void contract_set_average();
    void contract_single();
    void contract_complete();
    void prune_to_k_neighbors();
    
    

    scalar set_avg(LLAMANode * a, LLAMANode * b);

    static const bool parent_comp(const std::pair<LLAMANode *, scalar> &p1,
                                  const std::pair<LLAMANode *, scalar> &p2);

    std::unordered_map<node_id_t, node_id_t> cc_parents;

    std::vector<LLAMANode *> active_nodes;

    std::vector<LLAMANode *> all_nodes;
    void init_all_nodes() {
        all_nodes.reserve(MAX_NODES);
        for (size_t i=0; i < MAX_NODES; i++) {
            all_nodes.push_back(new LLAMANode(i));
        }
    }

    // per round
    // number of nodes
    std::vector<size_t> number_of_active_ids;

    // all_active_ids[r][i] gives the id of the ith node in round r
    std::vector<node_id_t *> all_active_ids;

    std::vector<std::unordered_map<node_id_t, node_id_t>> all_map_active_ids_to_seq_id;

    // all_parent2children[r][i] gives the children of the ith node in round r
    std::vector<std::vector<node_id_t> *> all_parent2children;

    // all_node2descendants[r][i] gives the descendants of the ith node in round r
    std::unordered_set<node_id_t> **all_node2descendants;
    size_t all_node2descendants_len = 0;

    void save_parents(std::unordered_set<LLAMANode * > &parent_ids);
    void set_descendants();
    void get_child_parent_edges();

    std::vector<node_id_t> descendants_r;
    std::vector<node_id_t> descendants_c;
    std::vector<node_id_t> children;
    std::vector<node_id_t> parents;

    ~LLAMA(){
        // delete [] all_nodes;
    };

    // timing methods
    std::chrono::time_point<std::chrono::high_resolution_clock> get_time()
    {
        return std::chrono::high_resolution_clock::now();
    }

    scalar sec(std::chrono::time_point<std::chrono::high_resolution_clock> st,
               std::chrono::time_point<std::chrono::high_resolution_clock> en)
    {
        return std::chrono::duration_cast<std::chrono::seconds>(en - st).count();
    }
};

#endif //_LLAMA_H
