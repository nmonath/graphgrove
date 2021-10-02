/**
 * Copyright (c) 2021 The authors of SCC All rights reserved.
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

# ifndef _SCC_H
# define _SCC_H

#include <atomic>
#include <fstream>
#include <iostream>
#include <stack>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <vector>
#include <shared_mutex>
#include <string>
#include <chrono>
#include <queue>
#include <bitset>
#include <chrono>


#ifdef __clang__
#define SHARED_MUTEX_TYPE shared_mutex
#else
#define SHARED_MUTEX_TYPE shared_timed_mutex
#endif

#include <Eigen/Core>

#include "utils.h"

typedef uint32_t node_id_t;

class SCC 
{
    public:

        // the current timestep in the universe
        int global_step = 0;
        // the number of levels of the tree
        unsigned num_levels;
        // thresholds used
        std::vector<scalar> thresholds;
        // number of parallel threads to use.
        unsigned cores = 1;

        const static unsigned ROTATE = 0;
        const static unsigned GRAFT = 1;
        const static unsigned FULL = 2;
        unsigned incremental_strategy = GRAFT; 

        const static unsigned NO_PRINT = 0;
        const static unsigned LEVEL_PRINT = 1;
        unsigned verbosity = NO_PRINT;

        const static unsigned SV = 0;
        const static unsigned FAST_SV = 1;

        unsigned cc_strategy = FAST_SV;
        unsigned par_cc_strategy = FAST_SV;
        size_t par_minimum = 0;
        scalar total_time = 0.0;

        // should we assume that datapoints
        // will be given in order 0,1,2,3,4, ...
        bool assume_level_zero_sequential = true;

        // stats
        scalar knn_time = 0.0;
        scalar update_time = 0.0;
        scalar center_update_time = 0.0;
        int get_total_number_marked();
        int get_max_number_marked();
        int get_total_number_of_nodes();
        int get_max_cc_iterations();
        int get_sum_cc_iterations();
        int get_sum_cc_edges();
        int get_sum_cc_nodes();
        
        // how should we label the nodes for updates
        void set_marking_strategy(unsigned strat);

        void fit();
        void fit_incremental();

        SCC(std::vector<scalar> & thresh, unsigned cores);
        SCC(std::vector<scalar> & thresh, unsigned cores, unsigned cc_alg, size_t par_min, unsigned verbosity_level);
        ~SCC();
    
        static SCC * init(std::vector<scalar> &thresh, unsigned cores);
        static SCC * init(std::vector<scalar> &thresh, unsigned cores, unsigned cc_alg, size_t par_min, unsigned verbosity_level);

        // add edges to the graph and update SCC
        bool insert_graph_mb(std::vector<uint32_t> & r, std::vector<uint32_t>  &c, std::vector<scalar> &s);

        // add edges to the graph 
        bool add_graph_edges_mb(std::vector<uint32_t> & r, std::vector<uint32_t>  &c, std::vector<scalar> &s);

        // take the added edges and update SCC
        bool fit_on_graph();

        // add the first set edges to the graph in large batch fashion
        void insert_first_batch(size_t n, std::vector<uint32_t> & r, std::vector<uint32_t>  &c, std::vector<scalar> &s);

        // remove the markers on updated nodes
        void clear_marked();

        // update the global timestep
        void set_level_global_step();

        // debug print
        void print_structure();

        // summary stats
        scalar get_graph_update_time() {
            scalar res = 0.0;
            for (TreeLevel * l :levels) {
                res += l->graph_update_time;
            }
            return res;
        }

        scalar get_overall_update_time() {
            scalar res = 0.0;
            for (TreeLevel * l :levels) {
                res += l->overall_update_time;
            }
            return res;
        }
        scalar get_best_neighbor_time() {
            scalar res = 0.0;
            for (TreeLevel * l :levels) {
                res += l->best_neighbor_time;
            }
            return res;
        }
        scalar get_cc_time() {
            scalar res = 0.0;
            for (TreeLevel * l :levels) {
                res += l->cc_time;
            }
            return res;
        }


    class TreeLevel
        {
        public:
            
            // the scc object
            SCC * scc;

            // the timestep
            int global_step;

            // the level threshold
            scalar threshold;

            // the lowest possible similarity value
            constexpr static scalar lowest_value = -10000.0;

            // multithreading
            unsigned cores = 4;

            // monitoring stats
            scalar knn_time = 0.0;
            scalar update_time = 0.0;
            scalar center_update_time = 0.0;
            scalar graph_update_time = 0.0;
            scalar cc_time = 0.0;
            scalar best_neighbor_time = 0.0;
            scalar overall_update_time = 0.0;
            scalar other_update_time = 0.0;
            scalar num_iterations_cc = 0.0;
            scalar num_cc_edges = 0.0;
            scalar num_cc_nodes = 0.0;

            // how to update in online setting
            const static unsigned JUST_SIB = 0;
            const static unsigned MY_BEST_NEIGHBOR = 1;
            const static unsigned ALL_BEST_NEIGHBORS = 2;
            unsigned marking_strategy = MY_BEST_NEIGHBOR;

            // the height of the level (0 = leaves, 1= parents of leaves, etc.)
            unsigned height = 0;

            void summary_message();
            
            class TreeNode
                {
                    public:
                        // CC neighbor edges

                        class TreeNodeSimComparison
                        {
                            public:
                                bool operator() (const std::pair<TreeLevel::TreeNode*, scalar> & a, const std::pair<TreeLevel::TreeNode*, scalar> & b)  {
                                    return a.second > b.second; 
                                };
                        };

                        std::priority_queue< std::pair<TreeLevel::TreeNode*, scalar>,
                                            std::vector<std::pair<TreeLevel::TreeNode*, scalar> >,
                                            TreeNodeSimComparison > sorted_neighbors;


                        scalar count;
                        std::unordered_set<TreeLevel::TreeNode*> cc_neighbors;
                        std::unordered_set<TreeLevel::TreeNode*> best_neighbors;
                        std::unordered_map<TreeLevel::TreeNode*, scalar> neigh;
                        std::unordered_map<node_id_t, TreeNode *> children;

                        pointType _p;                       // point associated with the node
                        pointType sum;                       // sum of points
                        scalar Z;                       // number of descendants 
                        pointType mean; // mean of points
                        scalar maxdistUP; // upper bound

                        TreeLevel * level;
                        std::string ext_prop;               // external encoded propertoes of current node

                        TreeNode* parent = NULL;
                        TreeNode* prev_parent = NULL;

                        node_id_t this_id;
                        node_id_t point_rep_id;

                        bool deleted = false;
                        bool kid_left = false;
                        int last_updated = 0;
                        int created_time = 0;
                        int marked_time = 0;
                        bool changed = false;
                        int cc_kid_changed = 0;
                        int cc_changed = 0;
                        bool created_now = true;

                        std::set<node_id_t> descendant_leafs;
                        int descendant_leaf_update_time = -1;

                        // one NN edge
                        TreeLevel::TreeNode * cc_neighbor = NULL;  
                        scalar cc_neighbor_score = lowest_value; 

                        TreeLevel::TreeNode * last_cc_neighbor = NULL;  
                        scalar last_cc_neighbor_score = lowest_value; 

                        TreeLevel::TreeNode * best_neighbor = NULL;  
                        scalar best_neighbor_score = lowest_value; 

                        TreeLevel::TreeNode * last_best_neighbor = NULL;  
                        scalar last_best_neighbor_score  = lowest_value; 

                        // CC f value
                        TreeLevel::TreeNode * f = NULL;
                        TreeLevel::TreeNode * fnext = NULL;
                        TreeLevel::TreeNode * fprev = NULL;

                        // the value of f that was there the last time we added a parent.
                        TreeLevel::TreeNode * last_parent = NULL;
                        TreeLevel::TreeNode * curr_cc_parent = NULL;

                        std::shared_timed_mutex mtx;

                        TreeNode(node_id_t id) {
                            this_id = id;
                            point_rep_id = id;
                            count = 0;
                        }
                        ~TreeNode();

                        TreeNode * fastforward_levels() {
                            // skip over singleton kids
                            #ifdef DEBUG_SCC
                            std::cout << "fastforward_levels - this level " << level->height << " id " << this_id << " children.size() " << children.size() << std::endl;
                            #endif
                            size_t num_kids = this->children.size();
                            if (level->height == 1 && num_kids != 0) {
                                return this;
                            } else if (level->height == 0) {
                                return this;
                            } else if (num_kids > 1) {
                                return this;
                            } else if (num_kids == 1){
                                for (const auto & c: this->children) {
                                    return c.second->fastforward_levels();
                                }
                            } 
                            return NULL;
                        }

                        scalar dist(const pointType& pp) const   
                        {
                            return -mean.dot(pp); 
                        }

                        scalar sim(const pointType& pp) const  
                        {
                            return mean.dot(pp);
                        }

                        void set_descendants() {
                            if (descendant_leaf_update_time < this->level->global_step && this->level->height > 0) {
                                this->descendant_leafs.clear();
                            }
                            if (this->children.size() > 0) {
                                // std::cout << "this node is " << this_id << std::endl;
                                if (descendant_leaf_update_time < this->level->global_step) {
                                    for (const auto & pair: children) {
                                        
                                        if (!pair.second->deleted) {
                                            // std::cout << "this node is " << this_id << " kid is " << pair.first << std::endl;
                                            pair.second->set_descendants();
                                        }
                                    }
                                    for (const auto & pair: children) {
                                        for (node_id_t d: pair.second->descendant_leafs) {
                                            if (!pair.second->deleted) {
                                                // std::cout << "this node is " << this_id << " kid is " << pair.first << " d " << d << std::endl;
                                                this->descendant_leafs.insert(d);
                                            }
                                        }
                                    }
                                }
                            }
                            descendant_leaf_update_time = this->level->global_step;
                        }

                        std::set<node_id_t> get_descendants() {
                            set_descendants();
                            return descendant_leafs;
                        }

                        void print_info() {
                            std::cout << "node  " << this_id << " level " << level->height << " .deleted " <<  deleted << " count " << count << " .best_neighbor " <<  best_neighbor << " .f "  << curr_cc_parent << " .last_parent " << last_parent << " last updated " <<  last_updated << " created time " <<  created_time << " global " << level->global_step << " ccn ";
                            for (TreeNode * ccn : cc_neighbors) {
                                std::cout << " " << ccn->this_id ;
                            }
                            std::cout << " best_neighbors ";
                            for (TreeNode * ccn : best_neighbors) {
                                std::cout << " " << ccn->this_id;
                            }
                            std::cout << " kids ";
                            for (const auto & pair : children) {
                                std::cout << " " << pair.first;
                            }
                            std::cout << " neighbors ";
                            for (const auto & pair : neigh) {
                                std::cout << " (" << pair.first->this_id << ", " << pair.second << ")" ;
                            }
                            std::cout << std::endl;
                        }        
            };

            // deconstructor
            ~TreeLevel();

            // determine which nodes should be marked for update in the first round.
            void mark_for_first_round();

            // mapping from nodes in the level to their offsets in the vector of nodes
            std::unordered_map<node_id_t, size_t> nodeid2index;
            std::vector<TreeNode *> nodes;
            std::vector<TreeNode *> marked_nodes;
            std::set<TreeNode *> marked_node_set;

            std::shared_timed_mutex mtx;

            // build level / find parents
            void compute();
            void compute_incremental();

            void build_nearest_neighbor_graph();
            void par_build_nearest_neighbor_graph();
            void build_nearest_neighbor_graph_incremental();
            void par_build_nearest_neighbor_graph_incremental();

            void connected_components_sv();
            void connected_components_incremental_sv();
            void connected_components_fast_sv();
            void connected_components_incremental_fast_sv();

            void par_connected_components_sv();
            void par_connected_components_incremental_sv();
            void par_connected_components_fast_sv();
            void par_connected_components_incremental_fast_sv();

            TreeLevel::TreeNode * get_or_create_node(node_id_t a);

            TreeLevel::TreeNode * get_node(node_id_t a) {
                return this->nodes[this->nodeid2index[a]];
            }

            TreeLevel(scalar thresh, unsigned num_cores) {
                this->threshold = thresh;
                this->cores = num_cores;
            } 

            static bool update_levels(TreeLevel * prev_level, scalar thresh, TreeLevel * next_level);
            static bool par_update_levels(TreeLevel * prev_level, scalar thresh, TreeLevel * next_level);
            static TreeLevel* from_previous(TreeLevel * prev_level, scalar next_thresh);
            static TreeLevel* par_from_previous(TreeLevel * prev_level, scalar next_thresh);
        };


        std::vector<TreeLevel::TreeNode*> minibatch_points;
        std::set<TreeLevel::TreeNode*> observed_and_not_fit_marked;
        std::vector<TreeLevel *> levels;
        TreeLevel::TreeNode * record_point(node_id_t uid);

};





#endif //_SCC_H

