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

#include "scc.h"

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                SCC Fit Methods                         *
 *                                                                        *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

/**
 * Perform batch setting fit. 
 */ 
void SCC::fit() {
    size_t i = 1;
    assert(levels.size() == 1);
    while (i <= num_levels) {
        auto st_fit = utils::get_time();

        if (verbosity == LEVEL_PRINT) {
            std::cout << "Level Start - ";
            levels[i-1]->summary_message();
        } 

        #ifdef DEBUG_SCC
        std::cout << "level i " << i << std::endl;
        std::cout << "compute... " << std::endl;
        #endif

        levels[i-1]->compute();
        
        #ifdef DEBUG_SCC
        std::cout << "compute... done!" << std::endl;
        #endif
        #ifdef DEBUG_SCC
        std::cout << "add level... " << std::endl;
        #endif    
        if (cores == 1 || levels[i-1]->nodes.size() < par_minimum) {
            levels.push_back(SCC::TreeLevel::from_previous( levels[i-1], thresholds[i]));
        } else {
            levels.push_back(SCC::TreeLevel::par_from_previous( levels[i-1], thresholds[i]));
        }
        #ifdef DEBUG_SCC
        std::cout << "add level... done!" << std::endl;
        #endif
        auto en_fit = utils::get_time();
        total_time += utils::timedur(st_fit,en_fit);

        if (verbosity == LEVEL_PRINT) {
            std::cout << "Level End - ";
            levels[i-1]->summary_message();
        }

        i += 1;
    }
}

/**
 * Perform incremental setting fit. 
 */ 
void SCC::fit_incremental() {

    size_t i = 1;
    levels[0]->mark_for_first_round();
    auto st_fit = utils::get_time();

    while (i <= num_levels) {

        if (verbosity == LEVEL_PRINT) {
            std::cout << "Level Start - ";
            levels[i-1]->summary_message();
        } 

        #ifdef TIME_SCC
        std::cout << "level i " << i << "threshold " << thresholds[i] << std::endl;
        std::cout << "compute... " << std::endl;
        #endif

        levels[i-1]->compute_incremental();
        #ifdef TIME_SCC
        std::cout << "compute... done!" << std::endl;
        #endif
        
        #ifdef TIME_SCC
        std::cout << "replace level... " << std::endl;
        auto st_up = utils::get_time();
        #endif

        bool change_made = true;
        if (cores == 1 || levels[i-1]->marked_nodes.size() < par_minimum) {
            change_made = SCC::TreeLevel::update_levels(levels[i-1], thresholds[i], levels[i]);
        } else {
            change_made = SCC::TreeLevel::par_update_levels(levels[i-1], thresholds[i], levels[i]);
        }

        #ifdef TIME_SCC
        auto en_up = utils::get_time();
        std::cout << "#time update... " << utils::timedur(st_up, en_up) << std::endl;
        #endif
        auto en_fit = utils::get_time();
        total_time += utils::timedur(st_fit,en_fit);
        #ifdef TIME_SCC
        std::cout << "replace level... done!" << std::endl;
        #endif
        if (verbosity == LEVEL_PRINT) {
            std::cout << "Level end - ";
            std::cout << " incremental (change_made =" << change_made << ")";
            levels[i-1]->summary_message();
        } 
        if (!change_made) {
            #ifdef DEBUG_SCC
            std::cout << "no change made fit incremental! done!" << std::endl;
            #endif
            break;
        }
        i += 1;
    }
    #ifdef TIME_SCC
    auto en_fit = utils::get_time();
    std::cout << "#time fit_incremental " << utils::timedur(st_fit, en_fit) << std::endl;
    #endif
}


/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                               Compute Level                            *
 *                                                                        *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

void SCC::TreeLevel::compute() {
    if (cores == 1 || nodes.size() < scc->par_minimum) {
        build_nearest_neighbor_graph();
    } else {
        par_build_nearest_neighbor_graph();
    }
    if (scc->verbosity == LEVEL_PRINT) {
        std::cout << "Level Finished Best NN - ";
        summary_message();
    } 
    if (cores == 1 || nodes.size() < scc->par_minimum) {
        if (scc->cc_strategy == scc->FAST_SV) {
            connected_components_fast_sv();
        } else {
            connected_components_sv();
        }
    } else {
        if (scc->cc_strategy == scc->FAST_SV) {
            par_connected_components_fast_sv();
        } else {
            par_connected_components_sv();
        }
    }
    if (scc->verbosity == LEVEL_PRINT) {
        std::cout << "Level Connected Components - ";
        summary_message();
    } 
}

void SCC::TreeLevel::compute_incremental() {
    #ifdef TIME_SCC
    std::cout << "#time start level " << height << " marked " << marked_nodes.size() << std::endl;
    #endif 

    auto st_build_nn_g = utils::get_time();
    build_nearest_neighbor_graph_incremental();
    auto en_build_nn_g = utils::get_time();
    #ifdef TIME_SCC
    std::cout << "#time build_nn " << utils::timedur(st_build_nn_g, en_build_nn_g) << std::endl;
    #endif 
    best_neighbor_time += utils::timedur(st_build_nn_g, en_build_nn_g);

    auto st_cc = utils::get_time();
    if (cores == 1 || marked_nodes.size() < scc->par_minimum) {
        if (scc->cc_strategy == scc->FAST_SV) {
            connected_components_incremental_fast_sv();
        } else {
            connected_components_incremental_sv();
        }
    } else {
        if (scc->cc_strategy == scc->FAST_SV) {
            par_connected_components_incremental_fast_sv();
        } else {
            par_connected_components_incremental_sv();
        }
    }
    auto en_cc = utils::get_time();
    #ifdef TIME_SCC
    std::cout << "#time cc " << utils::timedur(st_cc, en_cc) << std::endl;
    #endif 
    cc_time += utils::timedur(st_cc, en_cc);
}


/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                        Build Neighbor Graph                            *
 *                                                                        *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

// building the nearest neighbor graph in batch setting.
void SCC::TreeLevel::build_nearest_neighbor_graph() { 
    auto st = utils::get_time();
    
    #ifdef DEBUG_SCC
    std::cout << "build_nearest_neighbor_graph - starting..." << std::endl;
    std::cout << "nodes.size() = " << nodes.size() << std::endl;
    #endif
    
    for (SCC::TreeLevel::TreeNode * u_node : nodes) {

        // node_id_t k = u_node->this_id;
        SCC::TreeLevel::TreeNode * best_neighbor = u_node;
        scalar best_val = lowest_value; 

        // loops over all the neighbors.
        for (const auto & pair : u_node->neigh) {
            SCC::TreeLevel::TreeNode * neigh_node = pair.first;
            if (neigh_node != u_node) {
                auto score = pair.second / (u_node->count * neigh_node->count);
                if (score > best_val && neigh_node != u_node) {
                    best_val = score;
                    best_neighbor = neigh_node;
                }
            }
        }

        if (best_val > this->threshold) {
            u_node->cc_neighbor = best_neighbor;
            u_node->cc_neighbor_score = best_val;
        } else {
            u_node->cc_neighbor = u_node;
            u_node->cc_neighbor_score = lowest_value;
        }
        u_node->best_neighbor = best_neighbor;
        u_node->best_neighbor_score = best_val;
        if (marking_strategy == ALL_BEST_NEIGHBORS) {
            u_node->best_neighbors.insert(best_neighbor);
            best_neighbor->best_neighbors.insert(u_node);
        }
        
    }
    auto en = utils::get_time();
    best_neighbor_time += utils::timedur(st,en);
    #ifdef DEBUG_SCC
    std::cout << "build_nearest_neighbor_graph - done in " << utils::timedur(st,en) << " seconds" << std::endl;
    #endif
}

void SCC::TreeLevel::par_build_nearest_neighbor_graph() { 
    auto st = utils::get_time();
    
    #ifdef DEBUG_SCC
    std::cout << "build_nearest_neighbor_graph - starting..." << std::endl;
    std::cout << "nodes.size() = " << nodes.size() << std::endl;
    #endif
    
    utils::parallel_for(0, nodes.size(), [&](size_t idx)->void{ 
        SCC::TreeLevel::TreeNode * u_node = nodes[idx];
        node_id_t k = u_node->this_id;
        SCC::TreeLevel::TreeNode * best_neighbor = u_node;
        scalar best_val = lowest_value; 

        // loops over all the neighbors.
        for (const auto & pair : u_node->neigh) {
            SCC::TreeLevel::TreeNode * neigh_node = pair.first;
            if (neigh_node != u_node) {
                auto score = pair.second / (u_node->count * neigh_node->count);
                if (score > best_val && neigh_node != u_node) {
                    best_val = score;
                    best_neighbor = neigh_node;
                }
            }
        }

        if (best_val > this->threshold) {
            u_node->cc_neighbor = best_neighbor;
            u_node->cc_neighbor_score = best_val;
        } else {
            u_node->cc_neighbor = u_node;
            u_node->cc_neighbor_score = lowest_value;
        }
        u_node->best_neighbor = best_neighbor;
        u_node->best_neighbor_score = best_val;
        if (marking_strategy == ALL_BEST_NEIGHBORS) {
            u_node->mtx.lock();
            u_node->best_neighbors.insert(best_neighbor);
            u_node->mtx.unlock();
            best_neighbor->mtx.lock();
            best_neighbor->best_neighbors.insert(u_node);
            best_neighbor->mtx.unlock();
        }
        
    }, cores);
    auto en = utils::get_time();
    best_neighbor_time += utils::timedur(st,en);
    #ifdef DEBUG_SCC
    std::cout << "build_nearest_neighbor_graph - done in " << utils::timedur(st,en) << " seconds" << std::endl;
    #endif
}

void SCC::TreeLevel::par_build_nearest_neighbor_graph_incremental() { 
    #ifdef DEBUG_SCC
    std::cout << "[best_nn] LEVEL " << height << " - build_nearest_neighbor_graph_incremental - starting..." << std::endl;
    std::cout << "[best_nn] LEVEL " << height << " - build_nearest_neighbor_graph_incremental - num nodes " << marked_nodes.size() << std::endl;
    #endif
    
    utils::parallel_for(0, marked_nodes.size(), [&](node_id_t idx)->void{
       
        SCC::TreeLevel::TreeNode * u_node = marked_nodes[idx];
        
        if (u_node->last_updated != global_step) {
            return;
        }

        SCC::TreeLevel::TreeNode * best_neighbor = u_node;        
        scalar best_val = lowest_value; 

        for (const auto & pair : u_node->neigh) {
            SCC::TreeLevel::TreeNode * neighbor_node = pair.first;
            if (!neighbor_node->deleted && neighbor_node->marked_time == u_node->marked_time) {
                if (pair.first != u_node) {
                    auto score = pair.second / (u_node->count * neighbor_node->count);
                    
                    #ifdef DEBUG_SCC
                    std::cout << "LEVEL " << height << " sim(" << k << ", " << pair.first << ") = " << score << std::endl;
                    #endif
                    
                    if (score > best_val && pair.first != u_node) {
                        best_val = score;
                        best_neighbor = neighbor_node;
                    }
                }
            }
        }

        u_node->last_cc_neighbor = u_node->cc_neighbor;
        u_node->last_cc_neighbor_score = u_node->cc_neighbor_score;
        u_node->last_best_neighbor = u_node->best_neighbor;
        u_node->best_neighbor = best_neighbor;
        u_node->best_neighbor_score = best_val;
        if (best_val > this->threshold) {
            u_node->cc_neighbor = best_neighbor;
            u_node->cc_neighbor_score = best_val;
        } else {
            u_node->cc_neighbor = u_node;
            u_node->cc_neighbor_score = lowest_value;
        }
        
        #ifdef DEBUG_SCC
        std::cout << "[best_nn] point " << k << " nodes[contig_k]->best_neighbor = " << nodes[contig_k]->best_neighbor << " nodes[contig_k]->last_best_neighbor  " << nodes[contig_k]->last_best_neighbor << std::endl;
        #endif
        
    }, cores);
    if (marking_strategy == ALL_BEST_NEIGHBORS) {
        utils::parallel_for(0, marked_nodes.size(), [&](node_id_t idx)->void{ 
            // uid of the nodes is k
            SCC::TreeLevel::TreeNode * u_node = marked_nodes[idx];
            // if we have a best neighbor and last best neighbor
            if (u_node->best_neighbor != NULL && u_node->last_best_neighbor != NULL) {
                // if we have changed our favorite neighbor 
                if (u_node->last_best_neighbor != u_node->best_neighbor) {
                    // remove last_best neighbor from cc edges unless they want to keep it
                    if (u_node->last_best_neighbor->best_neighbor != u_node) {
                        u_node->mtx.lock();
                        u_node->best_neighbors.erase(u_node->last_best_neighbor);
                        u_node->mtx.unlock();
                        u_node->last_best_neighbor->mtx.lock();
                        u_node->last_best_neighbor->best_neighbors.erase(u_node);
                        u_node->last_best_neighbor->mtx.unlock();
                    }
                }
            }
            if (u_node->best_neighbor != u_node && marking_strategy == ALL_BEST_NEIGHBORS) {
                u_node->mtx.lock();
                u_node->best_neighbors.insert(u_node->best_neighbor);
                u_node->mtx.unlock();
                u_node->best_neighbor->mtx.lock();
                u_node->best_neighbor->best_neighbors.insert(u_node);
                u_node->best_neighbor->mtx.unlock();
            }
            #ifdef DEBUG_SCC
            std::cout << "[best_nn] LEVEL " << height << " point " << u_node << " best_neighbor = " << u_node->best_neighbor << " last_best_neighbor  " << u_node->last_best_neighbor << " cc neighbors: ";
            for (node_id_t ccn: u_node->cc_neighbors) {
                std::cout << " " << ccn;
            }
            std::cout << " best neighbors ";
            for (node_id_t ccn: u_node->best_neighbors) {
                std::cout << " " << ccn;
            }
            std::cout << std::endl;
            #endif
        }, cores);
    }
    #ifdef DEBUG_SCC
    auto en = utils::get_time();
    std::cout << "build_nearest_neighbor_graph - done in " << utils::timedur(st,en) << " seconds" << std::endl;
    #endif
}


void SCC::TreeLevel::build_nearest_neighbor_graph_incremental() { 
    #ifdef DEBUG_SCC
    std::cout << "[best_nn] LEVEL " << height << " - build_nearest_neighbor_graph_incremental - starting..." << std::endl;
    std::cout << "[best_nn] LEVEL " << height << " - build_nearest_neighbor_graph_incremental - num nodes " << marked_nodes.size() << std::endl;
    #endif
    
    for (SCC::TreeLevel::TreeNode * u_node: marked_nodes) {
        
        if (u_node->last_updated != global_step) {
            continue;
        }

        SCC::TreeLevel::TreeNode * best_neighbor = u_node;        
        scalar best_val = lowest_value; 

        for (const auto & pair : u_node->neigh) {
            SCC::TreeLevel::TreeNode * neighbor_node = pair.first;
            if (!neighbor_node->deleted && neighbor_node->marked_time == u_node->marked_time) {
                if (pair.first != u_node) {
                    auto score = pair.second / (u_node->count * neighbor_node->count);
                    
                    #ifdef DEBUG_SCC
                    std::cout << "LEVEL " << height << " sim(" << k << ", " << pair.first << ") = " << score << std::endl;
                    #endif
                    
                    if (score > best_val && pair.first != u_node) {
                        best_val = score;
                        best_neighbor = neighbor_node;
                    }
                }
            }
        }

        // #ifdef DEBUG_SCC
        // std::cout << "point " << k << " best_idx = " << best_idx << " val " << best_val << " nodeid2index " << nodeid2index[best_idx] << std::endl;
        // #endif
        u_node->last_cc_neighbor = u_node->cc_neighbor;
        u_node->last_cc_neighbor_score = u_node->cc_neighbor_score;
        u_node->last_best_neighbor = u_node->best_neighbor;
        u_node->best_neighbor = best_neighbor;
        u_node->best_neighbor_score = best_val;
        if (best_val > this->threshold) {
            u_node->cc_neighbor = best_neighbor;
            u_node->cc_neighbor_score = best_val;
        } else {
            u_node->cc_neighbor = u_node;
            u_node->cc_neighbor_score = lowest_value;
        }
        
        #ifdef DEBUG_SCC
        std::cout << "[best_nn] point " << k << " nodes[contig_k]->best_neighbor = " << nodes[contig_k]->best_neighbor << " nodes[contig_k]->last_best_neighbor  " << nodes[contig_k]->last_best_neighbor << std::endl;
        #endif
        
    }

    if (marking_strategy == ALL_BEST_NEIGHBORS) {
       for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
            // if we have a best neighbor and last best neighbor
            if (u_node->best_neighbor != NULL && u_node->last_best_neighbor != NULL) {
                // if we have changed our favorite neighbor 
                if (u_node->last_best_neighbor != u_node->best_neighbor) {
                    // remove last_best neighbor from cc edges unless they want to keep it
                    if (u_node->last_best_neighbor->best_neighbor != u_node) {
                        u_node->best_neighbors.erase(u_node->last_best_neighbor);
                        u_node->last_best_neighbor->best_neighbors.erase(u_node);
                    }
                }
            }
            if (u_node->best_neighbor != u_node && marking_strategy == ALL_BEST_NEIGHBORS) {
                u_node->best_neighbors.insert(u_node->best_neighbor);
                u_node->best_neighbor->best_neighbors.insert(u_node);
            }
            #ifdef DEBUG_SCC
            std::cout << "[best_nn] LEVEL " << height << " point " << u_node << " best_neighbor = " << u_node->best_neighbor << " last_best_neighbor  " << u_node->last_best_neighbor << " cc neighbors: ";
            for (node_id_t ccn: u_node->cc_neighbors) {
                std::cout << " " << ccn;
            }
            std::cout << " best neighbors ";
            for (node_id_t ccn: u_node->best_neighbors) {
                std::cout << " " << ccn;
            }
            std::cout << std::endl;
            #endif
        }
    }
    #ifdef DEBUG_SCC
    auto en = utils::get_time();
    std::cout << "build_nearest_neighbor_graph - done in " << utils::timedur(st,en) << " seconds" << std::endl;
    #endif
}


/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                        CC Computation                                  *
 *                                                                        *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

void SCC::TreeLevel::connected_components_sv() 
{
    auto st = utils::get_time();
    #ifdef DEBUG_SCC
    std::cout << "connected_components - starting..." << std::endl;
    std::cout << "connected_components - num nodes... " << nodes.size() << std::endl;
    #endif
    bool converged = false;
    size_t num_iter = 0;

    for (SCC::TreeLevel::TreeNode * u_node : nodes) {
        u_node->f  = u_node;
        u_node->fprev = u_node;
        u_node->fnext = u_node;
    }

    while (!converged) {
        
        for (SCC::TreeLevel::TreeNode * u_node : nodes) {
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && v_node != u_node) {

                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_u = u_node->f;
                SCC::TreeLevel::TreeNode * f_f_u = f_u->f;
                if (f_u->this_id == f_f_u->this_id && f_v->this_id < f_u->this_id) {
                    f_u->fnext = f_v;
                }
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_u = u_node->f;
                f_f_u = f_u->f;
                if (f_u->this_id == f_f_u->this_id && f_v->this_id < f_u->this_id) {
                    f_u->fnext = f_v;
                }
            }
        }
        // std::cout << "SV algorithm....iter... " << num_iter << " tree hook done." << std::endl;

        for (SCC::TreeLevel::TreeNode * u_node : nodes) {
            u_node->f = u_node->fnext;
        }

        // std::cout << "SV algorithm....iter... " << num_iter << " fnext copy done." << std::endl;

        for (SCC::TreeLevel::TreeNode * u_node: nodes) {
            SCC::TreeLevel::TreeNode * f_u = u_node->f;
            SCC::TreeLevel::TreeNode * f_f_u = f_u->f;
            if (f_u != f_f_u) {
                u_node->fnext = f_f_u;
            }
        }
        // std::cout << "SV algorithm....iter... " << num_iter << " shortcut done." << std::endl;

        int changes = 0;
        for (SCC::TreeLevel::TreeNode * u_node : nodes) {
            u_node->f = u_node->fnext;
            u_node->curr_cc_parent = u_node->f;
            if (u_node->f != u_node->fprev) {
                // std::cout << "uidx: " << uidx << " f: " << nodes[uidx]->f << " fprev: " << nodes[uidx]->fprev << std::endl;
                changes++;
                u_node->fprev = u_node->f;
            }
        }

        // std::cout << "SV algorithm....iter... " << num_iter << " fnext copy done." << std::endl;
        converged = changes == 0;
        num_iter += 1;
    }
    
    // std::cout << "Final predicted components...." << std::endl;
    // print_all_cc_labels();
    auto en = utils::get_time();
    cc_time += utils::timedur(st, en);
    #ifdef DEBUG_SCC
    std::cout << "connected_components - done in " << utils::timedur(st, en) << " seconds." << std::endl;
    #endif
    num_iterations_cc = num_iter;
}


void SCC::TreeLevel::par_connected_components_sv() 
{
    auto st = utils::get_time();
    #ifdef DEBUG_SCC
    std::cout << "connected_components - starting..." << std::endl;
    std::cout << "connected_components - num nodes... " << nodes.size() << std::endl;
    #endif
    bool converged = false;
    size_t num_iter = 0;

    utils::parallel_for(0, nodes.size(),[&](size_t uidx)->void{
        SCC::TreeLevel::TreeNode * u_node = nodes[uidx];
        u_node->f  = u_node;
        u_node->fprev = u_node;
        u_node->fnext = u_node;
    }, cores);

    while (!converged) {

        utils::parallel_for(0, nodes.size(),[&](node_id_t uidx)->void{
            SCC::TreeLevel::TreeNode * u_node = nodes[uidx];
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && v_node != u_node) {
                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_u = u_node->f;
                SCC::TreeLevel::TreeNode * f_f_u = f_u->f;
                if (f_u->this_id == f_f_u->this_id && f_v->this_id < f_u->this_id) {
                    f_u->mtx.lock();
                    f_u->fnext = f_v;
                    f_u->mtx.unlock();
                }
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_u = u_node->f;
                f_f_u = f_u->f;
                if (f_u->this_id == f_f_u->this_id && f_v->this_id < f_u->this_id) {
                    f_u->mtx.lock();
                    f_u->fnext = f_v;
                    f_u->mtx.unlock();
                }
            }
        }, cores);
        // std::cout << "SV algorithm....iter... " << num_iter << " tree hook done." << std::endl;

        utils::parallel_for(0, nodes.size(),[&](node_id_t midx)->void{
            SCC::TreeLevel::TreeNode * u_node = nodes[midx];
            u_node->f = u_node->fnext;
        }, cores);

        // std::cout << "SV algorithm....iter... " << num_iter << " fnext copy done." << std::endl;

        utils::parallel_for(0, nodes.size(),[&](node_id_t uidx)->void{
            SCC::TreeLevel::TreeNode * u_node = nodes[uidx];
            SCC::TreeLevel::TreeNode * f_u = u_node->f;
            SCC::TreeLevel::TreeNode * f_f_u = f_u->f;
            u_node->fnext = f_f_u;
        }, cores);
        // std::cout << "SV algorithm....iter... " << num_iter << " shortcut done." << std::endl;

        std::atomic<long> changes(0);
        utils::parallel_for(0, nodes.size(),[&](node_id_t midx)->void{
            SCC::TreeLevel::TreeNode * u_node = nodes[midx];
            u_node->f = u_node->fnext;
            u_node->curr_cc_parent = u_node->f;
            if (u_node->f != u_node->fprev) {
                // std::cout << "uidx: " << uidx << " f: " << nodes[uidx]->f << " fprev: " << nodes[uidx]->fprev << std::endl;
                changes++;
                u_node->fprev = u_node->f;
            }
        }, cores);

        // std::cout << "SV algorithm....iter... " << num_iter << " fnext copy done." << std::endl;

        converged = changes == 0;
        
        num_iter += 1;
    }
    // std::cout << "Final predicted components...." << std::endl;
    // print_all_cc_labels();
    auto en = utils::get_time();
    cc_time += utils::timedur(st, en);
    #ifdef DEBUG_SCC
    std::cout << "connected_components - done in " << utils::timedur(st, en) << " seconds." << std::endl;
    #endif
    num_iterations_cc = num_iter;
}

void SCC::TreeLevel::connected_components_fast_sv() 
{
    auto st = utils::get_time();
    #ifdef DEBUG_SCC
    std::cout << "connected_components - starting..." << std::endl;
    std::cout << "connected_components - num nodes... " << nodes.size() << std::endl;
    #endif
    bool converged = false;
    size_t num_iter = 0;

    for (TreeNode * u_node: nodes) {
        u_node->f  = u_node;
        u_node->fprev = u_node;
        u_node->fnext = u_node;
    }

    while (!converged) {

        for (SCC::TreeLevel::TreeNode * u_node: nodes) {
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && v_node != u_node) {
                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_u = u_node->f;
                SCC::TreeLevel::TreeNode * f_f_v = f_v->f;
                if (f_f_v->this_id < f_u->fnext->this_id) {
                    f_u->fnext = f_f_v;
                } 
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_u = u_node->f;
                f_f_v = f_v->f;
                if (f_f_v->this_id < f_u->fnext->this_id) {
                    f_u->fnext = f_f_v;
                }
            }
        }

        for (SCC::TreeLevel::TreeNode * u_node: nodes) {
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && v_node != u_node) {
                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_f_v = f_v->f;
                if (f_f_v->this_id < u_node->fnext->this_id) {
                    u_node->fnext = f_f_v;
                } 
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_f_v = f_v->f;
                if (f_f_v->this_id < u_node->fnext->this_id) {
                    u_node->fnext = f_f_v;
                }
            }
        }

        // std::cout << "SV algorithm....iter... " << num_iter << " tree hook done." << std::endl;

        for (SCC::TreeLevel::TreeNode * u_node: nodes) {
            if (u_node->f->f->this_id < u_node->fnext->this_id) {
                u_node->fnext = u_node->f->f;
            }
        }

        long changes = 0;
        for (SCC::TreeLevel::TreeNode * u_node: nodes) {
            u_node->f = u_node->fnext;
            u_node->curr_cc_parent = u_node->f;
            if (u_node->f != u_node->fprev) {
                u_node->fprev = u_node->f;
                changes++;
            }
        }
        converged = changes == 0;
        num_iter += 1;
    }
    // std::cout << "Final predicted components...." << std::endl;
    // print_all_cc_labels();
    auto en = utils::get_time();
    cc_time += utils::timedur(st, en);
    #ifdef DEBUG_SCC
    std::cout << "connected_components - done in " << utils::timedur(st, en) << " seconds." << std::endl;
    #endif
    num_iterations_cc = num_iter;
}

void SCC::TreeLevel::par_connected_components_fast_sv() 
{
    auto st = utils::get_time();
    #ifdef DEBUG_SCC
    std::cout << "connected_components - starting..." << std::endl;
    std::cout << "connected_components - num nodes... " << nodes.size() << std::endl;
    #endif
    bool converged = false;
    size_t num_iter = 0;

    utils::parallel_for(0, nodes.size(),[&](size_t uidx)->void{
        SCC::TreeLevel::TreeNode * u_node = nodes[uidx];
        u_node->f  = u_node;
        u_node->fprev = u_node;
        u_node->fnext = u_node;
    }, cores);

    while (!converged) {

        utils::parallel_for(0, nodes.size(),[&](node_id_t uidx)->void{
            SCC::TreeLevel::TreeNode * u_node = nodes[uidx];
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && v_node != u_node) {
                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_u = u_node->f;
                SCC::TreeLevel::TreeNode * f_f_v = f_v->f;
                f_u->mtx.lock();
                if (f_f_v->this_id < f_u->fnext->this_id) {
                    f_u->fnext = f_f_v;
                } 
                f_u->mtx.unlock();
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_u = u_node->f;
                f_f_v = f_v->f;
                f_u->mtx.lock();
                if (f_f_v->this_id < f_u->fnext->this_id) {
                    f_u->fnext = f_f_v;
                }
                f_u->mtx.unlock();
            }
        }, cores);


        utils::parallel_for(0, nodes.size(),[&](node_id_t uidx)->void{
            SCC::TreeLevel::TreeNode * u_node = nodes[uidx];
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && v_node != u_node) {
                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_f_v = f_v->f;
                u_node->mtx.lock();
                if (f_f_v->this_id < u_node->fnext->this_id) {
                    u_node->fnext = f_f_v;
                } 
                u_node->mtx.unlock();
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_f_v = f_v->f;
                u_node->mtx.lock();
                if (f_f_v->this_id < u_node->fnext->this_id) {
                    u_node->fnext = f_f_v;
                }
                u_node->mtx.unlock();
            }
        }, cores);

        // std::cout << "SV algorithm....iter... " << num_iter << " tree hook done." << std::endl;

        utils::parallel_for(0, nodes.size(),[&](node_id_t midx)->void{
            SCC::TreeLevel::TreeNode * u_node = nodes[midx];
            if (u_node->f->f->this_id < u_node->fnext->this_id) {
                u_node->fnext = u_node->f->f;
            }
        }, cores);

        std::atomic<long> changes(0);
        utils::parallel_for(0, nodes.size(),[&](node_id_t midx)->void{
            SCC::TreeLevel::TreeNode * u_node = nodes[midx];
            u_node->f = u_node->fnext;
            u_node->curr_cc_parent = u_node->f;
            if (u_node->f != u_node->fprev) {
                u_node->fprev = u_node->f;
                changes++;
            }
        }, cores);

        converged = changes == 0;
        num_iter += 1;
    }
    auto en = utils::get_time();
    cc_time += utils::timedur(st, en);
    #ifdef DEBUG_SCC
    std::cout << "connected_components - done in " << utils::timedur(st, en) << " seconds." << std::endl;
    #endif
    num_iterations_cc = num_iter;
}

void SCC::TreeLevel::connected_components_incremental_sv() 
{
    #ifdef DEBUG_SCC
    auto st = utils::get_time();
    std::cout << "[cc] connected_components_incremental - starting..." << std::endl;
    std::cout << "[cc] connected_components_incremental - number of nodes ... " << marked_nodes.size() << std::endl;
    #endif
    bool converged = false;
    size_t num_iter = 0;

    // clear last round
    for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
        u_node->last_parent = u_node->curr_cc_parent;
        u_node->f = u_node;
        u_node->fprev = u_node;
        u_node->fnext = u_node;
    }

    while (!converged) {
        
        for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && u_node != v_node) {
                // only consider the edges between two things that are marked
                // and that are not deleted!
                if (v_node->marked_time != u_node->marked_time || u_node->marked_time == -1 || v_node->marked_time  == -1) {
                    continue;
                }
                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_u = u_node->f;
                SCC::TreeLevel::TreeNode * f_f_u = f_u->f;
                if (f_u->this_id == f_f_u->this_id && f_v->this_id < f_u->this_id) {
                    f_u->fnext = f_v;
                }
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_u = u_node->f;
                f_f_u = f_u->f;
                if (f_u->this_id == f_f_u->this_id && f_v->this_id < f_u->this_id) {
                    f_u->fnext = f_v;
                }
            }
        }

        // std::cout << "SV algorithm....iter... " << num_iter << " tree hook done." << std::endl;

        for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
            u_node->f = u_node->fnext;
        }

        // std::cout << "SV algorithm....iter... " << num_iter << " fnext copy done." << std::endl;

        for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
            SCC::TreeLevel::TreeNode * f_u = u_node->f;
            SCC::TreeLevel::TreeNode * f_f_u = f_u->f;
            u_node->fnext = f_f_u;
        }
        // std::cout << "SV algorithm....iter... " << num_iter << " shortcut done." << std::endl;

        long changes = 0;
        for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
            u_node->f = u_node->fnext;
            u_node->curr_cc_parent = u_node->f;
            if (u_node->f != u_node->fprev) {
                u_node->fprev = u_node->f;
                changes++;
            }
        }

        // std::cout << "SV algorithm....iter... " << num_iter << " fnext copy done." << std::endl;
        // std::cout << "SV algorithm....iter... " << num_iter << " number of changes " << changes << std::endl;

        converged = changes == 0;
        // print_all_cc_labels();
        num_iter += 1;
    }

    // std::cout << "Final predicted components...." << std::endl;
    // print_all_cc_labels();

    #ifdef DEBUG_SCC
    auto en = utils::get_time();
    std::cout << "connected_components_incremental - done in " << utils::timedur(st, en) << " seconds." << std::endl;
    #endif

    num_iterations_cc = num_iter;
    // return false;
}

void SCC::TreeLevel::par_connected_components_incremental_sv() 
{
    #ifdef DEBUG_SCC
    auto st = utils::get_time();
    std::cout << "[cc] connected_components_incremental - starting..." << std::endl;
    std::cout << "[cc] connected_components_incremental - number of nodes ... " << marked_nodes.size() << std::endl;
    #endif
    bool converged = false;
    size_t num_iter = 0;

    // clear last round
    utils::parallel_for(0, marked_nodes.size(),[&](size_t nidx)->void{
        SCC::TreeLevel::TreeNode * u_node = marked_nodes[nidx];
        u_node->last_parent = u_node->curr_cc_parent;
        u_node->f = u_node;
        u_node->fprev = u_node;
        u_node->fnext = u_node;
    }, cores);

    while (!converged) {
        utils::parallel_for(0, marked_nodes.size(),[&](size_t uidx)->void{
            SCC::TreeLevel::TreeNode * u_node = marked_nodes[uidx];
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && u_node != v_node) {
                // only consider the edges between two things that are marked
                // and that are not deleted!
                if (v_node->marked_time != u_node->marked_time || u_node->marked_time == -1 || v_node->marked_time  == -1) {
                    return;
                }
                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_u = u_node->f;
                SCC::TreeLevel::TreeNode * f_f_u = f_u->f;
                if (f_u->this_id == f_f_u->this_id && f_v->this_id < f_u->this_id) {
                    f_u->mtx.lock();
                    f_u->fnext = f_v;
                    f_u->mtx.unlock();
                }
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_u = u_node->f;
                f_f_u = f_u->f;
                if (f_u->this_id == f_f_u->this_id && f_v->this_id < f_u->this_id) {
                    f_u->mtx.lock();
                    f_u->fnext = f_v;
                    f_u->mtx.unlock();
                }
            }
        }, cores);

        // std::cout << "SV algorithm....iter... " << num_iter << " tree hook done." << std::endl;

        utils::parallel_for(0, marked_nodes.size(),[&](size_t midx)->void{
            SCC::TreeLevel::TreeNode * u_node = marked_nodes[midx];
            u_node->f = u_node->fnext;
        }, cores);

        // std::cout << "SV algorithm....iter... " << num_iter << " fnext copy done." << std::endl;

        utils::parallel_for(0, marked_nodes.size(),[&](size_t uidx)->void{
            SCC::TreeLevel::TreeNode * u_node = marked_nodes[uidx];
            SCC::TreeLevel::TreeNode * f_u = u_node->f;
            SCC::TreeLevel::TreeNode * f_f_u = f_u->f;
            u_node->fnext = f_f_u;
        }, cores);
        // std::cout << "SV algorithm....iter... " << num_iter << " shortcut done." << std::endl;

        std::atomic<long> changes(0);
        utils::parallel_for(0, marked_nodes.size(),[&](size_t midx)->void{
            SCC::TreeLevel::TreeNode * u_node = marked_nodes[midx];
            u_node->f = u_node->fnext;
            u_node->curr_cc_parent = u_node->f;
            if (u_node->f != u_node->fprev) {
                u_node->fprev = u_node->f;
                changes++;
            }
        }, cores);

        // std::cout << "SV algorithm....iter... " << num_iter << " fnext copy done." << std::endl;
        // std::cout << "SV algorithm....iter... " << num_iter << " number of changes " << changes << std::endl;

        converged = changes == 0;
        
        num_iter += 1;
    }

    // std::cout << "Final predicted components...." << std::endl;
    // print_all_cc_labels();
    #ifdef DEBUG_SCC
    auto en = utils::get_time();
    std::cout << "connected_components_incremental - done in " << utils::timedur(st, en) << " seconds." << std::endl;
    #endif

    num_iterations_cc = num_iter;
}

void SCC::TreeLevel::connected_components_incremental_fast_sv()  {
    #ifdef DEBUG_SCC
    auto st = utils::get_time();
    std::cout << "[cc] connected_components_incremental - starting..." << std::endl;
    std::cout << "[cc] connected_components_incremental - number of nodes ... " << marked_nodes.size() << std::endl;
    #endif
    bool converged = false;
    size_t num_iter = 0;

    // who needs an update?
    num_cc_edges = 0;
    // clear last round
    for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
        u_node->last_parent = u_node->f;
        u_node->f = u_node;
        u_node->fprev = u_node;
        u_node->fnext = u_node;
        // if (u_node->cc_neighbor != u_node) {
        //     num_cc_edges += 1;
        // }
    }

    while (!converged) {
   
        for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && v_node != u_node) {
                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_u = u_node->f;
                SCC::TreeLevel::TreeNode * f_f_v = f_v->f;
                if (f_f_v->this_id < f_u->fnext->this_id) {
                    f_u->fnext = f_f_v;
                } 
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_u = u_node->f;
                f_f_v = f_v->f;
                if (f_f_v->this_id < f_u->fnext->this_id) {
                    f_u->fnext = f_f_v;
                }
            }
        }

        // std::cout << "SV algorithm....iter... " << num_iter << " tree hook done." << std::endl;

        for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && v_node != u_node) {
                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_f_v = f_v->f;
                if (f_f_v->this_id < u_node->fnext->this_id) {
                    u_node->fnext = f_f_v;
                } 
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_f_v = f_v->f;
                if (f_f_v->this_id < u_node->fnext->this_id) {
                    u_node->fnext = f_f_v;
                }
            }
        }

        // std::cout << "SV algorithm....iter... " << num_iter << " tree hook done." << std::endl;

        for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
            if (u_node->f->f->this_id < u_node->fnext->this_id) {
                u_node->fnext =u_node->f->f;
            }
        }

        // check convergence
        long changes = 0;
        for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
            u_node->f = u_node->fnext;
            u_node->curr_cc_parent = u_node->f;
            if (u_node->f != u_node->fprev) {
                u_node->fprev = u_node->f;
                changes++;
            }
        }

        converged = changes == 0;
        num_iter += 1;
    }


    #ifdef DEBUG_SCC
    auto en = utils::get_time();
    std::cout << "connected_components_incremental - done in " << utils::timedur(st, en) << " seconds." << std::endl;
    #endif

    num_iterations_cc = num_iter;
}

void SCC::TreeLevel::par_connected_components_incremental_fast_sv() 
{
    #ifdef DEBUG_SCC
    auto st = utils::get_time();
    std::cout << "[cc] connected_components_incremental - starting..." << std::endl;
    std::cout << "[cc] connected_components_incremental - number of nodes ... " << marked_nodes.size() << std::endl;
    #endif
    bool converged = false;
    size_t num_iter = 0;

    num_cc_edges = 0;
    // clear last round
    utils::parallel_for(0, marked_nodes.size(),[&](size_t nidx)->void{
        SCC::TreeLevel::TreeNode * u_node = marked_nodes[nidx];
        u_node->last_parent = u_node->curr_cc_parent;
        u_node->f = u_node;
        u_node->fprev = u_node;
        u_node->fnext = u_node;
    }, cores);

    while (!converged) {
   
        utils::parallel_for(0, marked_nodes.size(),[&](size_t uidx)->void{
            SCC::TreeLevel::TreeNode * u_node = marked_nodes[uidx];
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && v_node != u_node) {
                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_u = u_node->f;
                SCC::TreeLevel::TreeNode * f_f_v = f_v->f;
                f_u->mtx.lock();
                if (f_f_v->this_id < f_u->fnext->this_id) {
                    f_u->fnext = f_f_v;
                } 
                f_u->mtx.unlock();
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_u = u_node->f;
                f_f_v = f_v->f;
                f_u->mtx.lock();
                if (f_f_v->this_id < f_u->fnext->this_id) {
                    f_u->fnext = f_f_v;
                }
                f_u->mtx.unlock();
            }
        }, cores);

        // std::cout << "SV algorithm....iter... " << num_iter << " tree hook done." << std::endl;

        utils::parallel_for(0, marked_nodes.size(),[&](node_id_t uidx)->void{
            SCC::TreeLevel::TreeNode * u_node = marked_nodes[uidx];
            SCC::TreeLevel::TreeNode * v_node = u_node->cc_neighbor;
            if (v_node != NULL && v_node != u_node) {
                SCC::TreeLevel::TreeNode * f_v = v_node->f;
                SCC::TreeLevel::TreeNode * f_f_v = f_v->f;
                u_node->mtx.lock();
                if (f_f_v->this_id < u_node->fnext->this_id) {
                    u_node->fnext = f_f_v;
                } 
                u_node->mtx.unlock();
                std::swap(u_node, v_node);
                f_v = v_node->f;
                f_f_v = f_v->f;
                u_node->mtx.lock();
                if (f_f_v->this_id < u_node->fnext->this_id) {
                    u_node->fnext = f_f_v;
                }
                u_node->mtx.unlock();
            }
        }, cores);

        // std::cout << "SV algorithm....iter... " << num_iter << " tree hook done." << std::endl;

        utils::parallel_for(0, marked_nodes.size(),[&](node_id_t midx)->void{
            SCC::TreeLevel::TreeNode * u_node = marked_nodes[midx];
            if (u_node->f->f->this_id < u_node->fnext->this_id) {
                u_node->fnext =u_node->f->f;
            }
        }, cores);

        std::atomic<long> changes(0);
        utils::parallel_for(0, marked_nodes.size(),[&](node_id_t midx)->void{
            SCC::TreeLevel::TreeNode * u_node = marked_nodes[midx];
            u_node->f = u_node->fnext;
            u_node->curr_cc_parent = u_node->f;
            if (u_node->f != u_node->fprev) {
                u_node->fprev = u_node->f;
                changes++;
            }
        }, cores);

        converged = changes == 0;
        num_iter += 1;
    }

    // std::cout << "Final predicted components...." << std::endl;
    // print_all_cc_labels();

    #ifdef DEBUG_SCC
    auto en = utils::get_time();
    std::cout << "connected_components_incremental - done in " << utils::timedur(st, en) << " seconds." << std::endl;
    #endif
    num_iterations_cc = num_iter;
}

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                       Form Next Level                                  *
 *                                                                        *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

SCC::TreeLevel* SCC::TreeLevel::from_previous(TreeLevel * prev_level, scalar thresh) {
    auto st_update = utils::get_time();
    TreeLevel * t = NULL;
    t = new TreeLevel(thresh, prev_level->cores);
    t->marking_strategy = prev_level->marking_strategy;
    t->global_step = prev_level->global_step;
    t->height = prev_level->height + 1;
    t->scc = prev_level->scc;

    #ifdef DEBUG_SCC
    std::cout << "build from previous round ... " << std::endl;
    #endif

    // Add the nodes for this next level
    for (SCC::TreeLevel::TreeNode * a  : prev_level->nodes) {
        SCC::TreeLevel::TreeNode * par = t->get_or_create_node(a->curr_cc_parent->this_id);
        par->mtx.lock();
        par->children[a->this_id] = a;
        par->mtx.unlock();
        a->parent = par;
    }
    
    float graph_update = 0.0f;
    float other_update = 0.0f;
    for (SCC::TreeLevel::TreeNode* u_node: t->nodes) {
        auto st_up = utils::get_time();
    
        u_node->last_updated = t->global_step;
        u_node->marked_time = t->global_step;

        auto en_up = utils::get_time();
        other_update += utils::timedur(st_up, en_up);

        for (const auto & kid_id_pair: u_node->children) {

            auto st_other = utils::get_time();
            SCC::TreeLevel::TreeNode * kid = kid_id_pair.second;
            #ifdef DEBUG_SCC
            std::cout << "updating " << u_node->this_id << " from kid " << kid->this_id << std::endl;
            u_node->print_info();
            #endif

            u_node->count += kid->count;

            auto en_other = utils::get_time();
            other_update += utils::timedur(st_other, en_other);

            auto st_graph = utils::get_time();
            for (const auto & neigh_pair: kid->neigh) {
                SCC::TreeLevel::TreeNode * neigh_node = neigh_pair.first;
                SCC::TreeLevel::TreeNode * neigh_par_node = neigh_node->parent;
                node_id_t neigh_par = neigh_par_node->this_id;
                if (!neigh_node->deleted && neigh_par != u_node->this_id) {
                    u_node->neigh[neigh_par_node] += neigh_pair.second;
                }
            }
            auto en_graph = utils::get_time();
            graph_update += utils::timedur(st_graph, en_graph);
                 
        }
    }
    auto en_update = utils::get_time();
    prev_level->graph_update_time += (float) graph_update; 
    prev_level->overall_update_time += utils::timedur(st_update, en_update);;
    return t;
}



SCC::TreeLevel* SCC::TreeLevel::par_from_previous(TreeLevel * prev_level, scalar thresh) {
    auto st_update = utils::get_time();
    TreeLevel * t = NULL;
    t = new TreeLevel(thresh, prev_level->cores);
    t->marking_strategy = prev_level->marking_strategy;
    t->global_step = prev_level->global_step;
    t->height = prev_level->height + 1;
    t->scc = prev_level->scc;

    #ifdef DEBUG_SCC
    std::cout << "build from previous round ... " << std::endl;
    #endif

    // Add the nodes for this next level
    utils::parallel_for(0, prev_level->nodes.size(),[&](node_id_t uidx)->void{
        SCC::TreeLevel::TreeNode * a = prev_level->nodes[uidx];
        SCC::TreeLevel::TreeNode * par = t->get_or_create_node(a->curr_cc_parent->this_id);
        par->mtx.lock();
        par->children[a->this_id] = a;
        par->mtx.unlock();
        a->parent = par;
    }, prev_level->cores);
    
    std::atomic<long> graph_update(0);
    std::atomic<long> other_update(0);
    utils::parallel_for(0, t->nodes.size(),[&](node_id_t uidx)->void{
        SCC::TreeLevel::TreeNode * u_node = t->nodes[uidx];
    
        u_node->last_updated = t->global_step;
        u_node->marked_time = t->global_step;

        for (const auto & kid_id_pair: u_node->children) {

            // auto st_other = utils::get_time();
            SCC::TreeLevel::TreeNode * kid = kid_id_pair.second;
            #ifdef DEBUG_SCC
            std::cout << "updating " << u_node->this_id << " from kid " << kid->this_id << std::endl;
            u_node->print_info();
            #endif

            u_node->count += kid->count;

            // auto en_other = utils::get_time();
            // other_update += utils::timedur(st_other, en_other);

            auto st_graph = utils::get_time();
            for (const auto & neigh_pair: kid->neigh) {
                SCC::TreeLevel::TreeNode * neigh_node = neigh_pair.first;
                SCC::TreeLevel::TreeNode * neigh_par_node = neigh_node->parent;
                node_id_t neigh_par = neigh_par_node->this_id;
                if (!neigh_node->deleted && neigh_par != u_node->this_id) {
                    u_node->neigh[neigh_par_node] += neigh_pair.second;
                }
            }
            auto en_graph = utils::get_time();
            graph_update += utils::timedur_long(st_graph, en_graph);      
        }
    }, prev_level->cores);
    auto en_update = utils::get_time();
    prev_level->graph_update_time += ((float) graph_update / (float) 1000000.0);
    prev_level->overall_update_time += utils::timedur(st_update, en_update);
    return t;
}


bool SCC::TreeLevel::update_levels(TreeLevel * prev_level, scalar thresh, TreeLevel * next_level) {
    auto st_update_level = utils::get_time();

    std::set<SCC::TreeLevel::TreeNode *> next_round_marked;
    std::set<SCC::TreeLevel::TreeNode *> old_parents;

    bool parents_need_update = false;
    bool someone_had_null_parent = false;

    next_level->marked_nodes.clear(); 

    for (SCC::TreeLevel::TreeNode * u_node: prev_level->marked_nodes) {

        #ifdef DEBUG_SCC
        std::cout << "[form next level] LEVEL " << prev_level->height << "node " << u_node->this_id << std::endl; 
        std::cout << "[form next level] LEVEL " << prev_level->height << "node " << u_node->this_id << " curr_cc_parent " << u_node->curr_cc_parent->this_id << std::endl; 
        #endif

        // set old parent
        u_node->prev_parent = u_node->parent;
        // get or create new parent
        u_node->parent = next_level->get_or_create_node(u_node->curr_cc_parent->this_id);
        // mark new parent
        next_round_marked.insert(u_node->parent);

        #ifdef DEBUG_SCC
        std::cout << "[form next level] LEVEL " << prev_level->height << "node " << u_node->this_id << " parent " << u_node->parent->this_id;
        if (u_node->prev_parent == NULL) {
            std::cout << " prev " << " NULL ";
        } else {
            std::cout << " prev " << u_node->prev_parent->this_id  << std::endl;
        }
        #endif

        // we can't do a lazy a update. we MUST update things.
        if (u_node->prev_parent == NULL || u_node->prev_parent->this_id != u_node->parent->this_id || u_node->parent->count == 0) {
            #ifdef DEBUG_SCC
            std::cout << "[form next level] parents_need_update " << true << std::endl;
            #endif
            parents_need_update = true;
        }
        if (u_node->prev_parent == NULL || u_node->parent->count == 0) {
            #ifdef DEBUG_SCC
            std::cout << "[form next level] parents_need_update " << true << std::endl;
            #endif
            someone_had_null_parent = true;
        }

        // mark the old parents' siblings
        if (u_node->prev_parent != NULL) {
            if (u_node->prev_parent->parent != NULL) {
                for (const auto & sib: u_node->prev_parent->parent->children) {
                    // first is the id of the node in the level
                    #ifdef DEBUG_SCC
                    std::cout << "[form next level] marking " << sib.first << " with parent " << u_node->prev_parent->this_id << " of " << u << std::endl;
                    #endif
                    next_round_marked.insert(sib.second);
                }
            }
            if (prev_level->marking_strategy == SCC::TreeLevel::MY_BEST_NEIGHBOR) {
                // also mark the best neighbor as an optimistic attempt
                if (u_node->prev_parent->best_neighbor != NULL && u_node->prev_parent->best_neighbor != u_node->prev_parent ) {
                    #ifdef DEBUG_SCC
                    std::cout << "[form next level] marking best neighbor " << u_node->prev_parent->best_neighbor << " of prev parent " << u_node->prev_parent->this_id << " of " << u << std::endl;
                    #endif
                    next_round_marked.insert(u_node->prev_parent->best_neighbor);
                }
            } else if (prev_level->marking_strategy == SCC::TreeLevel::ALL_BEST_NEIGHBORS) {
                for (SCC::TreeLevel::TreeNode * m: u_node->prev_parent->best_neighbors) {
                    if (m != u_node->prev_parent ) {
                        #ifdef DEBUG_SCC
                        std::cout << "[form next level] marking best neighbor " << u_node->prev_parent->best_neighbor << " of prev parent " << u_node->prev_parent->this_id << " of " << u << std::endl;
                        #endif
                        next_round_marked.insert(m);
                    }
                }
            }
        }

        // remove me from old parents kids 
        if (u_node->prev_parent != NULL && u_node->prev_parent != u_node->parent) {
            u_node->prev_parent->children.erase(u_node->this_id);
            #ifdef DEBUG_SCC
            std::cout << "[form next level] removing " << u << " from " << " of prev parent " << u_node->prev_parent->this_id << " of " << u << " parent has number of kids: " << u_node->prev_parent->children.size() << std::endl;
            #endif
            if (u_node->prev_parent->children.empty()) {
                old_parents.insert(u_node->prev_parent);
            }
            u_node->prev_parent->kid_left = true;
            // u_node->prev_parent->last_updated = t->global_step;
        } 

        // add me to the new parent's children
        u_node->parent->children[u_node->this_id] = u_node;
        #ifdef DEBUG_SCC
        std::cout << "[form next level] adding " << u << " to " << " parent " << u_node->parent->this_id << " with number of kids:" << u_node->parent->children.size() << std::endl;
        #endif
    }

    // find the parents that have no more children.
    std::unordered_set<SCC::TreeLevel::TreeNode*> to_delete_level;
    for (SCC::TreeLevel::TreeNode* o_node: old_parents) {
        if (o_node->children.empty()) {
            o_node->deleted = true;
            to_delete_level.insert(o_node);
            #ifdef DEBUG_SCC
            std::cout << "[form next level] no more kids for " << o << std::endl;
            #endif
            next_round_marked.erase(o_node);
        }
    }
    #ifdef TIME_SCC
    auto st_deletion = utils::get_time();
    #endif
    std::unordered_set<SCC::TreeLevel::TreeNode*> to_delete_next;
    while (!to_delete_level.empty()) {

        #ifdef DEBUG_SCC
        std::cout << "to_delete_level.size() " << to_delete_level.size() << std::endl; 
        #endif

        // remove all deleted from parent's children
        for (SCC::TreeLevel::TreeNode* node: to_delete_level) {
            if (node->parent != NULL) {
                #ifdef DEBUG_SCC
                std::cout << "delete " << node->this_id << " level " << node->level->height << " from " << node->parent->this_id << " at level " << node->parent->level->height << std::endl;
                #endif
                node->parent->children.erase(node->this_id);
            }
        }
        
        std::set<node_id_t> parents_we_have;
        to_delete_next.clear();
        for (SCC::TreeLevel::TreeNode* node: to_delete_level) {
            if (node->parent != NULL && node->parent->children.empty()) {
                #ifdef DEBUG_SCC
                std::cout << "node " << node->this_id << " level " << node->level->height << " kids " << node->children.size() << std::endl; 
                std::cout << "par " << node->parent->this_id << " level " << node->parent->level->height << " kids " << node->parent->children.size() << std::endl; 
                #endif
                node->count = 0;
                node->deleted = true;
                node->cc_neighbors.clear();
                node->best_neighbors.clear();
                node->neigh.clear();
                node->marked_time = -1;

                if (node->sum.size() != 0) {
                    node->sum.setZero();
                    node->mean.setZero();
                    node->mean.resize(0);
                    node->sum.resize(0);
                }

                to_delete_next.insert(node->parent);
                node->parent = NULL;
            } else if (node->parent != NULL) {
                SCC::TreeLevel::TreeNode * par = node->parent;
                #ifdef DEBUG_SCC
                std::cout << "node " << node->this_id << " level " << node->level->height << " kids " << node->children.size() << std::endl; 
                std::cout << "par " << node->parent->this_id << " level " << node->parent->level->height << " kids " << node->parent->children.size() << std::endl; 
                #endif
                par->count -= node->count;
                node->count = 0;
                node->deleted = true;
                node->cc_neighbors.clear();
                node->best_neighbors.clear();
                node->neigh.clear();
                node->marked_time = 0;
                node->parent = NULL;
            } else {
                #ifdef DEBUG_SCC
                std::cout << "node " << node->this_id << " level " << node->level->height << " kids " << node->children.size() << std::endl; 
                bool isnull = node->parent == NULL;
                std::cout << "par is NULL " << isnull  << std::endl; 
                #endif
                node->count = 0;
                node->deleted = true;
                node->cc_neighbors.clear();
                node->best_neighbors.clear();
                node->neigh.clear();
                node->marked_time = 0;
                node->parent = NULL;
            }
        }
        std::swap(to_delete_next,to_delete_level);
    }  
    #ifdef TIME_SCC
    auto en_deletion = utils::get_time();
    std::cout << "#time deletion " << utils::timedur(st_deletion, en_deletion) << std::endl;
    #endif

    // lazy update the tree structure.
    #ifdef DEBUG_SCC
    std::cout << "parents_need_update " << parents_need_update << std::endl;
    #endif
    if (!parents_need_update) {  
        return false;
    }

    // short cut! we don't want to pay for entire update. just setting of counts.
    if (!someone_had_null_parent) {
        return false;
    }

    std::vector<SCC::TreeLevel::TreeNode*> to_update;
    next_level->marked_nodes.clear();
    for (SCC::TreeLevel::TreeNode * n: next_round_marked) {
        if (!n->deleted) {
            n->marked_time = next_level->global_step;
            to_update.push_back(n);
            next_level->marked_nodes.push_back(n);
            #ifdef DEBUG_SCC
            std::cout << "going to update / mark " << m << " deleted " << n->deleted << " #kids " << n->children.size() << " last_updated " << n->last_updated << std::endl;
            #endif
        }
    }

    scalar graph_update = 0.0f;
    scalar vector_update = 0.0f;
    scalar other_update = 0.0f;
    for (SCC::TreeLevel::TreeNode * u_node: to_update) {
        auto st_up = utils::get_time();
        // look at your kids
        // if none have been updated then we can skip this update
        bool skip_update = true;
        if (!u_node->kid_left && u_node->count > 0) {
            for (const auto & cpair: u_node->children) {
                SCC::TreeLevel::TreeNode* c = cpair.second;
                if (c->prev_parent == NULL || c->prev_parent->this_id != u_node->this_id || c->last_updated == next_level->global_step) {
                    #ifdef DEBUG_SCC
                    std::cout << "need to update " << u_node->this_id << std::endl; 
                    #endif
                    skip_update = false;
                    break;
                }
            }
        } else {
            skip_update = false;
        }

        if (skip_update) {
            continue;
        }

        u_node->count = 0;
        u_node->deleted = false;
        u_node->kid_left = false;
        u_node->neigh.clear();
        u_node->cc_neighbors.clear();
        u_node->best_neighbors.clear();
        u_node->last_updated = next_level->global_step;
        u_node->marked_time = next_level->global_step;
        u_node->sum.setZero();
        u_node->mean.setZero();
        auto en_up = utils::get_time();
        other_update += utils::timedur(st_up, en_up);
        for (const auto & kid_id_pair: u_node->children) {

            auto st_other = utils::get_time();
            SCC::TreeLevel::TreeNode * kid = kid_id_pair.second;
            #ifdef DEBUG_SCC
            std::cout << "updating " << u_node->this_id << " from kid " << kid->this_id << std::endl;
            u_node->print_info();
            #endif

            u_node->count += kid->count;

            auto en_other = utils::get_time();
            other_update += utils::timedur(st_other, en_other);

            auto st_graph = utils::get_time();

            for (const auto & neigh_pair: kid->neigh) {
                SCC::TreeLevel::TreeNode * neigh_node = neigh_pair.first;
                SCC::TreeLevel::TreeNode * neigh_par_node = neigh_node->parent;
                node_id_t neigh_par = neigh_par_node->this_id;
                if (!neigh_node->deleted && neigh_par != u_node->this_id) {
                    u_node->neigh[neigh_par_node] += neigh_pair.second;
                }
            }
            auto en_graph = utils::get_time();
            graph_update += utils::timedur(st_graph, en_graph);
        }
    }
    prev_level->graph_update_time += graph_update;

    #ifdef TIME_SCC
    std::cout << "#time graph_update " << graph_update << std::endl;
    std::cout << "#time vector_update " << vector_update << std::endl;
    std::cout << "#time other_update " << other_update << std::endl;
    #endif

    #ifdef DEBUG_SCC
    std::cout << "symmetry! " << std::endl;
    #endif
    // auto st_symmetrize = utils::get_time();
    // symmetrize
    // for (SCC::TreeLevel::TreeNode* u_node: to_update) {
    //     for (const auto & neigh_pair: u_node->neigh) {
    //         SCC::TreeLevel::TreeNode * neigh_node = t->get_node(neigh_pair.first);
    //         #ifdef DEBUG_SCC
    //         std::cout << "using " << u_node->this_id << " to update " << neigh_pair.first << std::endl;
    //         neigh_node->print_info();
    //         #endif
    //         if (!neigh_node->deleted) {
    //             neigh_node->neigh[u_node->this_id] = neigh_pair.second;
    //         }
    //         neigh_node->last_updated = t->global_step;
    //         #ifdef DEBUG_SCC
    //         std::cout << "using " << u_node->this_id << " to update " << neigh_pair.first << std::endl;
    //         neigh_node->print_info();
    //         #endif
    //     }
    // }
    auto en_update_level = utils::get_time();
    prev_level->overall_update_time += utils::timedur(st_update_level, en_update_level);

    #ifdef TIME_SCC
    std::cout << "#time symmetrize " << utils::timedur(st_symmetrize, en_update_level) << std::endl;
    std::cout << "[form next level] LEVEL " << prev_level->height << " from_previous to "  << next_level->height << " done!" << std::endl;
    std::cout << "#time update_level " << utils::timedur(st, en_update_level) << std::endl;
    #endif
    return true;
}

bool SCC::TreeLevel::par_update_levels(TreeLevel * prev_level, scalar thresh, TreeLevel * next_level) {
    auto st_update_level = utils::get_time();

    std::set<SCC::TreeLevel::TreeNode *> next_round_marked;
    std::set<SCC::TreeLevel::TreeNode *> old_parents;

    bool parents_need_update = false;
    bool someone_had_null_parent = false;

    next_level->marked_nodes.clear(); 

    utils::parallel_for(0, prev_level->marked_nodes.size(), [&](node_id_t idx)->void{ 
        SCC::TreeLevel::TreeNode * u_node = prev_level->marked_nodes[idx];

        #ifdef DEBUG_SCC
        std::cout << "[form next level] LEVEL " << prev_level->height << "node " << u_node->this_id << std::endl; 
        std::cout << "[form next level] LEVEL " << prev_level->height << "node " << u_node->this_id << " curr_cc_parent " << u_node->curr_cc_parent->this_id << std::endl; 
        #endif
        std::set<SCC::TreeLevel::TreeNode *> next_round_marked_local;
        std::set<SCC::TreeLevel::TreeNode *> old_parents_local;


        // set old parent
        u_node->prev_parent = u_node->parent;
        // get or create new parent
        u_node->parent = next_level->get_or_create_node(u_node->curr_cc_parent->this_id);
        // mark new parent
        next_round_marked_local.insert(u_node->parent);

        #ifdef DEBUG_SCC
        std::cout << "[form next level] LEVEL " << prev_level->height << "node " << u_node->this_id << " parent " << u_node->parent->this_id;
        if (u_node->prev_parent == NULL) {
            std::cout << " prev " << " NULL ";
        } else {
            std::cout << " prev " << u_node->prev_parent->this_id  << std::endl;
        }
        #endif

        // we can't do a lazy a update. we MUST update things.
        if (u_node->prev_parent == NULL || u_node->prev_parent->this_id != u_node->parent->this_id || u_node->parent->count == 0) {
            #ifdef DEBUG_SCC
            std::cout << "[form next level] parents_need_update " << true << std::endl;
            #endif
            parents_need_update = true;
        }
        if (u_node->prev_parent == NULL || u_node->parent->count == 0) {
            #ifdef DEBUG_SCC
            std::cout << "[form next level] parents_need_update " << true << std::endl;
            #endif
            someone_had_null_parent = true;
        }

        // mark the old parents' siblings
        if (u_node->prev_parent != NULL) {
            if (u_node->prev_parent->parent != NULL) {
                for (const auto & sib: u_node->prev_parent->parent->children) {
                    // first is the id of the node in the level
                    #ifdef DEBUG_SCC
                    std::cout << "[form next level] marking " << sib.first << " with parent " << u_node->prev_parent->this_id << " of " << u << std::endl;
                    #endif
                    next_round_marked_local.insert(sib.second);
                }
            }
            if (prev_level->marking_strategy == SCC::TreeLevel::MY_BEST_NEIGHBOR) {
                // also mark the best neighbor as an optimistic attempt
                if (u_node->prev_parent->best_neighbor != NULL && u_node->prev_parent->best_neighbor != u_node->prev_parent ) {
                    #ifdef DEBUG_SCC
                    std::cout << "[form next level] marking best neighbor " << u_node->prev_parent->best_neighbor << " of prev parent " << u_node->prev_parent->this_id << " of " << u << std::endl;
                    #endif
                    next_round_marked_local.insert(u_node->prev_parent->best_neighbor);
                }
            } else if (prev_level->marking_strategy == SCC::TreeLevel::ALL_BEST_NEIGHBORS) {
                for (SCC::TreeLevel::TreeNode * m: u_node->prev_parent->best_neighbors) {
                    if (m != u_node->prev_parent ) {
                        #ifdef DEBUG_SCC
                        std::cout << "[form next level] marking best neighbor " << u_node->prev_parent->best_neighbor << " of prev parent " << u_node->prev_parent->this_id << " of " << u << std::endl;
                        #endif
                        next_round_marked_local.insert(m);
                    }
                }
            }
        }

        // remove me from old parents kids 
        if (u_node->prev_parent != NULL && u_node->prev_parent != u_node->parent) {
            u_node->prev_parent->mtx.lock();
            u_node->prev_parent->children.erase(u_node->this_id);
            #ifdef DEBUG_SCC
            std::cout << "[form next level] removing " << u << " from " << " of prev parent " << u_node->prev_parent->this_id << " of " << u << " parent has number of kids: " << u_node->prev_parent->children.size() << std::endl;
            #endif
            if (u_node->prev_parent->children.empty()) {
                old_parents_local.insert(u_node->prev_parent);
            }
            u_node->prev_parent->kid_left = true;
            // u_node->prev_parent->last_updated = t->global_step;
            u_node->prev_parent->mtx.unlock();
        } 

        // add me to the new parent's children
        u_node->parent->mtx.lock();
        u_node->parent->children[u_node->this_id] = u_node;
        #ifdef DEBUG_SCC
        std::cout << "[form next level] adding " << u << " to " << " parent " << u_node->parent->this_id << " with number of kids:" << u_node->parent->children.size() << std::endl;
        #endif
        u_node->parent->mtx.unlock();

        prev_level->mtx.lock();
        next_round_marked.insert(next_round_marked_local.begin(), next_round_marked_local.end());
        old_parents.insert(old_parents_local.begin(), old_parents_local.end());
        prev_level->mtx.unlock();

    }, prev_level->cores);

    // find the parents that have no more children.
    std::unordered_set<SCC::TreeLevel::TreeNode*> to_delete_level;
    for (SCC::TreeLevel::TreeNode* o_node: old_parents) {
        if (o_node->children.empty()) {
            o_node->deleted = true;
            to_delete_level.insert(o_node);
            #ifdef DEBUG_SCC
            std::cout << "[form next level] no more kids for " << o << std::endl;
            #endif
            next_round_marked.erase(o_node);
        }
    }
    #ifdef TIME_SCC
    auto st_deletion = utils::get_time();
    #endif
    std::unordered_set<SCC::TreeLevel::TreeNode*> to_delete_next;
    while (!to_delete_level.empty()) {

        #ifdef DEBUG_SCC
        std::cout << "to_delete_level.size() " << to_delete_level.size() << std::endl; 
        #endif

        // remove all deleted from parent's children
        for (SCC::TreeLevel::TreeNode* node: to_delete_level) {
            if (node->parent != NULL) {
                #ifdef DEBUG_SCC
                std::cout << "delete " << node->this_id << " level " << node->level->height << " from " << node->parent->this_id << " at level " << node->parent->level->height << std::endl;
                #endif
                node->parent->children.erase(node->this_id);
            }
        }
        
        std::set<node_id_t> parents_we_have;
        to_delete_next.clear();
        for (SCC::TreeLevel::TreeNode* node: to_delete_level) {
            if (node->parent != NULL && node->parent->children.empty()) {
                #ifdef DEBUG_SCC
                std::cout << "node " << node->this_id << " level " << node->level->height << " kids " << node->children.size() << std::endl; 
                std::cout << "par " << node->parent->this_id << " level " << node->parent->level->height << " kids " << node->parent->children.size() << std::endl; 
                #endif
                node->count = 0;
                node->deleted = true;
                node->cc_neighbors.clear();
                node->best_neighbors.clear();
                node->neigh.clear();
                node->marked_time = -1;

                if (node->sum.size() != 0) {
                    node->sum.setZero();
                    node->mean.setZero();
                    node->mean.resize(0);
                    node->sum.resize(0);
                }
                to_delete_next.insert(node->parent);
                node->parent = NULL;
            } else if (node->parent != NULL) {
                SCC::TreeLevel::TreeNode * par = node->parent;
                #ifdef DEBUG_SCC
                std::cout << "node " << node->this_id << " level " << node->level->height << " kids " << node->children.size() << std::endl; 
                std::cout << "par " << node->parent->this_id << " level " << node->parent->level->height << " kids " << node->parent->children.size() << std::endl; 
                #endif
                par->count -= node->count;
                node->count = 0;
                node->deleted = true;
                node->cc_neighbors.clear();
                node->best_neighbors.clear();
                node->neigh.clear();
                node->marked_time = 0;
                if (node->sum.size() != 0) {
                    node->parent->sum -= node->sum;
                    node->parent->mean = node->parent->sum / ((scalar)par->count);
                }
                node->parent = NULL;
            } else {
                #ifdef DEBUG_SCC
                std::cout << "node " << node->this_id << " level " << node->level->height << " kids " << node->children.size() << std::endl; 
                bool isnull = node->parent == NULL;
                std::cout << "par is NULL " << isnull  << std::endl; 
                #endif
                node->count = 0;
                node->deleted = true;
                node->cc_neighbors.clear();
                node->best_neighbors.clear();
                node->neigh.clear();
                node->marked_time = 0;
                node->parent = NULL;
            }
        }
        std::swap(to_delete_next,to_delete_level);
    }  
    #ifdef TIME_SCC
    auto en_deletion = utils::get_time();
    std::cout << "#time deletion " << utils::timedur(st_deletion, en_deletion) << std::endl;
    #endif

    // lazy update the tree structure.
    #ifdef DEBUG_SCC
    std::cout << "parents_need_update " << parents_need_update << std::endl;
    #endif
    if (!parents_need_update) {  
        return false;
    }

    // short cut! we don't want to pay for entire update. just setting of counts.
    if (!someone_had_null_parent) {
        return false;
    }

    std::vector<SCC::TreeLevel::TreeNode*> to_update;
    next_level->marked_nodes.clear();
    for (SCC::TreeLevel::TreeNode * n: next_round_marked) {
        if (!n->deleted) {
            n->marked_time = next_level->global_step;
            to_update.push_back(n);
            next_level->marked_nodes.push_back(n);
            #ifdef DEBUG_SCC
            std::cout << "going to update / mark " << m << " deleted " << n->deleted << " #kids " << n->children.size() << " last_updated " << n->last_updated << std::endl;
            #endif
        }
    }


    std::atomic<long> graph_update(0);
    std::atomic<long> vector_update(0);
    std::atomic<long> other_update(0);
    utils::parallel_for(0, to_update.size(), [&](node_id_t idx)->void{ 
        SCC::TreeLevel::TreeNode * u_node = to_update[idx];
        auto st_up = utils::get_time();
        // look at your kids
        // if none have been updated then we can skip this update
        bool skip_update = true;
        if (!u_node->kid_left && u_node->count > 0) {
            for (const auto & cpair: u_node->children) {
                SCC::TreeLevel::TreeNode* c = cpair.second;
                if (c->prev_parent == NULL || c->prev_parent->this_id != u_node->this_id || c->last_updated == next_level->global_step) {
                    #ifdef DEBUG_SCC
                    std::cout << "need to update " << u_node->this_id << std::endl; 
                    #endif
                    skip_update = false;
                    break;
                }
            }
        } else {
            skip_update = false;
        }

        if (skip_update) {
            return;
        }

        u_node->count = 0;
        u_node->deleted = false;
        u_node->kid_left = false;
        u_node->neigh.clear();
        u_node->cc_neighbors.clear();
        u_node->best_neighbors.clear();
        u_node->last_updated = next_level->global_step;
        u_node->marked_time = next_level->global_step;
        u_node->sum.setZero();
        u_node->mean.setZero();
        auto en_up = utils::get_time();
        other_update += utils::timedur(st_up, en_up);
        for (const auto & kid_id_pair: u_node->children) {

            auto st_other = utils::get_time();
            SCC::TreeLevel::TreeNode * kid = kid_id_pair.second;
            #ifdef DEBUG_SCC
            std::cout << "updating " << u_node->this_id << " from kid " << kid->this_id << std::endl;
            u_node->print_info();
            #endif

            u_node->count += kid->count;

            auto en_other = utils::get_time();
            other_update += utils::timedur(st_other, en_other);

            auto st_graph = utils::get_time();

            for (const auto & neigh_pair: kid->neigh) {
                SCC::TreeLevel::TreeNode * neigh_node = neigh_pair.first;
                SCC::TreeLevel::TreeNode * neigh_par_node = neigh_node->parent;
                node_id_t neigh_par = neigh_par_node->this_id;
                if (!neigh_node->deleted && neigh_par != u_node->this_id) {
                    u_node->neigh[neigh_par_node] += neigh_pair.second;
                }
            }

            auto en_graph = utils::get_time();
            graph_update += utils::timedur_long(st_graph, en_graph);
        }
    }, prev_level->cores);

    #ifdef TIME_SCC
    std::cout << "#time graph_update " << graph_update << std::endl;
    std::cout << "#time vector_update " << vector_update << std::endl;
    std::cout << "#time other_update " << other_update << std::endl;
    #endif

    #ifdef DEBUG_SCC
    std::cout << "symmetry! " << std::endl;
    #endif
    // auto st_symmetrize = utils::get_time();
    // symmetrize
    // for (SCC::TreeLevel::TreeNode* u_node: to_update) {
    //     for (const auto & neigh_pair: u_node->neigh) {
    //         SCC::TreeLevel::TreeNode * neigh_node = t->get_node(neigh_pair.first);
    //         #ifdef DEBUG_SCC
    //         std::cout << "using " << u_node->this_id << " to update " << neigh_pair.first << std::endl;
    //         neigh_node->print_info();
    //         #endif
    //         if (!neigh_node->deleted) {
    //             neigh_node->neigh[u_node->this_id] = neigh_pair.second;
    //         }
    //         neigh_node->last_updated = t->global_step;
    //         #ifdef DEBUG_SCC
    //         std::cout << "using " << u_node->this_id << " to update " << neigh_pair.first << std::endl;
    //         neigh_node->print_info();
    //         #endif
    //     }
    // }
    auto en_update_level = utils::get_time();
    prev_level->overall_update_time += utils::timedur(st_update_level, en_update_level);
    prev_level->graph_update_time += (float) graph_update / (float) 1000000.0;
    #ifdef TIME_SCC
    std::cout << "#time symmetrize " << utils::timedur(st_symmetrize, en_update_level) << std::endl;
    std::cout << "[form next level] LEVEL " << prev_level->height << " from_previous to "  << next_level->height << " done!" << std::endl;
    std::cout << "#time update_level " << utils::timedur(st, en_update_level) << std::endl;
    #endif
    return true;
}

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                Constructors                            *
 *                                                                        *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

SCC * SCC::init(std::vector<scalar> &thresh, unsigned cores) {
    std::cout << "SCC.init v024" << std::endl;
    SCC* scc = NULL;
    scc = new SCC(thresh, cores);
    TreeLevel *round0 = NULL;
    round0 = new TreeLevel(thresh[0], cores);
    round0->scc = scc;
    scc->levels.push_back(round0);
    return scc;
}

SCC * SCC::init(std::vector<scalar> &thresh, unsigned cores, unsigned cc_alg, size_t par_min, unsigned verbosity_level) {
    std::cout << "SCC.init v024" << std::endl;
    SCC* scc = NULL;
    scc = new SCC(thresh, cores, cc_alg, par_min, verbosity_level);
    TreeLevel *round0 = NULL;
    round0 = new TreeLevel(thresh[0], cores);
    round0->scc = scc;
    scc->levels.push_back(round0);
    return scc;
}

SCC::TreeLevel::TreeNode * SCC::TreeLevel::get_or_create_node(node_id_t a) {
    mtx.lock_shared();
    size_t idx = nodes.size();
    TreeNode * new_node = NULL;
    #ifdef DEBUG_SCC
    std::cout << "get_or_create_node(" << a << ")" << std::endl; 
    #endif
    if (nodeid2index.find(a) == nodeid2index.end()) {
        mtx.unlock_shared();
        mtx.lock();
        // we create it
        if (nodeid2index.find(a) == nodeid2index.end()) {
            idx = nodes.size();
            new_node = new TreeNode(a);
            nodes.push_back(new_node); 
            nodes[idx]->level = this;
            nodes[idx]->deleted = false;
            nodes[idx]->count = 0;
            nodeid2index[a] = idx;
            nodes[idx]->created_time = global_step;
            nodes[idx]->last_updated = global_step;
            nodes[idx]->marked_time = global_step;
            #ifdef DEBUG_SCC
            std::cout << "a " << a << " idx " << nodeid2index[a] << std::endl;
            #endif
        } else { // someone else created it.
            new_node = nodes[nodeid2index[a]];
        }
        mtx.unlock();
    } else {
        new_node = nodes[nodeid2index[a]];
        if (new_node->deleted) {
            mtx.unlock_shared();
            new_node->mtx.lock();
            #ifdef DEBUG_SCC
            std::cout << "undeleted! " << a << " idx " << nodeid2index[a] << std::endl;
            #endif
            new_node->deleted = false;
            new_node->created_time = global_step;
            new_node->last_updated = global_step;
            new_node->marked_time = global_step;
            new_node->mtx.unlock();
        } else {
            mtx.unlock_shared();
        }
    }
    // auto en = utils::get_time();
    // #ifdef DEBUG_SCC
    // std::cout << "add_to_level_init - done " << utils::timedur(st,en) << " seconds." << std::endl;
    // #endif
    return new_node;
}

SCC::~SCC() {
    // std::cout << "SCC deconstructor start" << std::endl;
    //  std::flush(std::cout);
    for (size_t idx=0; idx < levels.size(); idx++) {
        // std::cout << "SCC deconstructor delete level " << idx << std::endl;
        //  std::flush(std::cout);
        delete levels[idx];
    }
    levels.clear();
    // std::cout << "SCC deconstructor end!" << std::endl;
    //  std::flush(std::cout);
}


SCC::TreeLevel::TreeNode::~TreeNode() {
    // std::cout << "node deconstructor start" << std::endl;
    //  std::flush(std::cout);
    neigh.clear();
    children.clear();
    best_neighbors.clear();
    // std::cout << "node deconstructor end" << std::endl;
    //  std::flush(std::cout);
}


SCC::TreeLevel::~TreeLevel() {
    // std::cout << "level deconstructor start" << std::endl;
    //  std::flush(std::cout);
    for (size_t idx=0; idx < nodes.size(); idx++) {
        // std::cout << "level deconstructor start delete node " << idx << std::endl;
        //  std::flush(std::cout);
        delete nodes[idx];
    }
    nodes.clear();
    // std::cout << "level deconstructor end!" << std::endl;
    //  std::flush(std::cout);
}

SCC::SCC(std::vector<scalar> &thresh, unsigned cores) {
    this->thresholds = thresh;
    this->num_levels = thresholds.size();
    this->cores = cores;
}

SCC::SCC(std::vector<scalar> &thresh, unsigned cores, unsigned cc_alg, size_t par_min, unsigned verbosity_level) {
    this->thresholds = thresh;
    this->num_levels = thresholds.size();
    this->cores = cores;
    this->cc_strategy = cc_alg;
    this->par_minimum = par_min;
    this->verbosity = verbosity_level;
}

void SCC::set_marking_strategy(unsigned strat) {
    for (TreeLevel * l: levels) {
        l->marking_strategy = strat;
    }
}

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                         Adding to base level                           *
 *                                                                        *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

SCC::TreeLevel::TreeNode * SCC::record_point(node_id_t uid) {
    // record the data point in the graph
    if (assume_level_zero_sequential) {
        if (uid >= levels[0]->nodes.size()) {
            levels[0]->nodes.reserve(uid);
            for (size_t i=levels[0]->nodes.size(); i <= uid; i++) {
                SCC::TreeLevel::TreeNode * n = new TreeLevel::TreeNode(i);
                levels[0]->nodes.push_back(n);
                // levels[0]->marked_nodes.push_back(n);
                levels[0]->nodeid2index[i] = i;
                n->level = levels[0];
                n->count = 1;
                n->Z = (scalar) 1.0;
                n->created_time = global_step;
                n->last_updated = global_step;
                n->descendant_leafs.insert(i);
            }
        }
        return levels[0]->nodes[uid];
    } else {
        if (levels[0]->nodeid2index.find(uid) == levels[0]->nodeid2index.end()) {
            #ifdef DEBUG_SCC
            std::cout << "record_point " << uid << std::endl;
            #endif
            #ifdef DEBUG_SCC
            std::cout << "levels[0].size() " << levels[0]->marked_nodes.size() << std::endl;
            #endif
            SCC::TreeLevel::TreeNode * n = new TreeLevel::TreeNode(uid);
            levels[0]->nodes.push_back(n);
            // levels[0]->marked_nodes.push_back(n);
            levels[0]->nodeid2index[uid] = levels[0]->nodes.size()-1;
            n->level = levels[0];
            n->count = 1;
            n->Z = (scalar) 1.0;
            n->created_time = global_step;
            n->last_updated = global_step;
            n->descendant_leafs.insert(uid);
            #ifdef DEBUG_SCC
            std::cout << "levels[0].size() " << levels[0]->marked_nodes.size() << std::endl;
            #endif
            return n;
        } else {
            return levels[0]->nodes[levels[0]->nodeid2index[uid]];
        }
    }
}

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                         Helper Methods                                 *
 *                                                                        *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

void SCC::clear_marked() {
    for (size_t l=0; l < levels.size(); l++) {
        levels[l]->marked_nodes.clear();
    }
}

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                         Insert Methods                                 *
 *                                                                        *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */


void SCC::insert_first_batch(size_t num_points, std::vector<uint32_t> & r, 
    std::vector<uint32_t>  &c, std::vector<scalar> &s) {
    TreeLevel *round0 = levels[0];
    auto st_knn = utils::get_time();
    round0->nodes.reserve(num_points);
    for (size_t i=0; i <= num_points; i++) {
        SCC::TreeLevel::TreeNode * n = new TreeLevel::TreeNode(i);
        levels[0]->nodes.push_back(n);
        // levels[0]->marked_nodes.push_back(n);
        levels[0]->nodeid2index[i] = i;
        n->level = levels[0];
        n->count = 1;
        n->Z = (scalar) 1.0;
        n->created_time = global_step;
        n->last_updated = global_step;
        n->descendant_leafs.insert(i);
    }
    if (cores == 1) {
        for (size_t i=0; i < r.size(); i++) {
             if (i % 100000 ==0) {
                std::cout << "\r Init " <<  i << " out of " << r.size() << "- " << (float) i*100.0 / (float) r.size() << "%" << " in " << utils::timedur(st_knn, utils::get_time()) << " seconds.";
            }
            SCC::TreeLevel::TreeNode* r_node = round0->nodes[r[i]]; 
            SCC::TreeLevel::TreeNode* c_node = round0->nodes[c[i]]; 
            r_node->neigh[c_node] = s[i];
            c_node->neigh[r_node] = s[i];
        }
    } else {
        utils::parallel_for(0, r.size(), [&](node_id_t i)->void{ 
            if (i % 100000 ==0) {
                round0->mtx.lock();
                std::cout << "\r Init " <<  i << " out of " << r.size() << "- " << (float) i*100.0 / (float) r.size() << "%" << " in " << utils::timedur(st_knn, utils::get_time()) << " seconds.";
                round0->mtx.unlock();
            }

            SCC::TreeLevel::TreeNode* r_node = round0->nodes[r[i]]; 
            SCC::TreeLevel::TreeNode* c_node = round0->nodes[c[i]]; 
            r_node->mtx.lock();
            r_node->neigh[c_node] = s[i];
            r_node->mtx.unlock();
            c_node->mtx.lock();
            c_node->neigh[r_node] = s[i];
            c_node->mtx.unlock();

        }, cores);
    }
    fit();
    global_step += 1;
}


/** 
 * Observe new edges, update SCC
 */
bool SCC::add_graph_edges_mb(std::vector<uint32_t> & r, 
    std::vector<uint32_t>  &c, std::vector<scalar> &s) {
    
    #ifdef DEBUG_SCC
    std::cout << "begin record --" <<std::endl;
    std::cout << "r.size() " << r.size() << std::endl;
    #endif    

    auto st_knn = utils::get_time();

    bool is_first_insert = levels.size() == 1;
    
    std::set<SCC::TreeLevel::TreeNode*> new_points;

    bool should_log = r.size() > 1000000;
    
    for (size_t i=0; i < r.size(); i++) {
        #ifdef DEBUG_SCC
        std::cout << "r " << r[i] << " c " << c[i] << " s " << s[i] << std::endl;
        #endif
        if (should_log) {
            if (i % 1000000 ==0) {
                std::cout << "\r Init " <<  i << " out of " << r.size() << "- " << (float) i*100.0 / (float) r.size() << "%" << " in " << utils::timedur(st_knn, utils::get_time()) << " seconds.";
            }
        }
        size_t num_pts = levels[0]->nodes.size();
        SCC::TreeLevel::TreeNode* r_node = record_point(r[i]);
        bool r_new = r_node->created_now;
        r_node->created_now = false;
        num_pts = levels[0]->nodes.size();
        SCC::TreeLevel::TreeNode* c_node = record_point(c[i]);
        bool c_new = c_node->created_now;
        c_node->created_now = false;

        r_node->neigh[c_node] = s[i];
        c_node->neigh[r_node] = s[i];
        r_node->last_updated = global_step;
        c_node->last_updated = global_step;

        if (!is_first_insert) {
            if (r_new) {
                new_points.insert(r_node);
                observed_and_not_fit_marked.insert(r_node);
            }
            if (c_new) {
                new_points.insert(c_node);
                observed_and_not_fit_marked.insert(c_node);
            }
            if (incremental_strategy == GRAFT) {
                observed_and_not_fit_marked.insert(r_node);
                r_node->last_updated = global_step;
                c_node->last_updated = global_step;
                r_node->marked_time = global_step;
                c_node->marked_time = global_step;
                observed_and_not_fit_marked.insert(c_node);
            }
        }

        #ifdef DEBUG_SCC
        std::cout << "record -- r_node " << r_node->this_id << " c_node " << c_node->this_id <<std::endl;
        #endif
    }
    if (!is_first_insert) {
        for (SCC::TreeLevel::TreeNode * m : new_points) {
            minibatch_points.push_back(m);
        }
    }

    auto en_knn = utils::get_time();
    knn_time += utils::timedur(st_knn, en_knn);

    #ifdef TIME_SCC
    std::cout << "end record --" <<std::endl;
    std::cout << "knn_time -- " << knn_time << " " << utils::timedur(st_knn, en_knn) << std::endl;
    #endif

    #ifdef DEBUG_SCC
    print_structure();
    #endif
    
    return true;
}

bool SCC::fit_on_graph() {
    
    set_level_global_step();
    
    auto st_knn = utils::get_time();

    TreeLevel *round0 = levels[0];

    clear_marked();
    for (SCC::TreeLevel::TreeNode * a: observed_and_not_fit_marked) {
        levels[0]->marked_nodes.push_back(a);
        levels[0]->marked_node_set.insert(a);
        a->last_updated = global_step;
    }
    auto en_knn = utils::get_time();
    knn_time += utils::timedur(st_knn, en_knn);

    #ifdef TIME_SCC
    std::cout << "end record --" <<std::endl;
    std::cout << "knn_time -- " << knn_time << " " << utils::timedur(st_knn, en_knn) << std::endl;
    #endif

    auto st_update = utils::get_time();
    #ifdef TIME_SCC
    // step 3: update tree
    std::cout << "start fit --" <<std::endl;
    #endif
    
    if (levels.size() == 1) {
        fit();
    } else{
        fit_incremental();
    }
    auto en_update = utils::get_time();
    update_time += utils::timedur(st_update, en_update);
    
    #ifdef TIME_SCC
    std::cout << "end time --" <<std::endl;
    std::cout << "update time -- " << update_time << " " << utils::timedur(st_update, en_update) << std::endl;
    #endif

    #ifdef DEBUG_SCC
    print_structure();
    #endif

    global_step += 1;
    minibatch_points.clear();
    observed_and_not_fit_marked.clear();
    return true;
}



bool SCC::insert_graph_mb(std::vector<uint32_t> & r,  std::vector<uint32_t>  &c, std::vector<scalar> &s) {
   add_graph_edges_mb(r, c, s);
   return fit_on_graph();
}


/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                         Counting & Timing                              *
 *                                                                        *
 ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */
int SCC::get_total_number_marked() {
    int total = 0;
    for (TreeLevel * l : levels ) {
        total += l->marked_nodes.size();
    }
    return total;
}

int SCC::get_total_number_of_nodes() {
    int total = 0;
    for (TreeLevel * l : levels ) {
        for (SCC::TreeLevel::TreeNode * n: l->nodes) {
            if (!n->deleted) {
                total+= 1;
            }
        }
    }
    return total;
}

int SCC::get_max_number_marked() {
    size_t max = 0;
    for (TreeLevel * l : levels ) {
        if (l->marked_nodes.size() > max) {
            max = l->marked_nodes.size();
        }
    }
    return (int) max;
}

int SCC::get_max_cc_iterations() {
    int max = 0;
    for (TreeLevel * l : levels ) {
        if (l->num_iterations_cc > max) {
            max = l->num_iterations_cc;
        }
    }
    return max;
}

int SCC::get_sum_cc_iterations() {
    int sum = 0;
    for (TreeLevel * l : levels ) {
        sum += l->num_iterations_cc;
    }
    return sum;
}

int SCC::get_sum_cc_edges() {
    int sum = 0;
    for (TreeLevel * l : levels ) {
        sum += l->num_cc_edges;
    }
    return sum;
}

int SCC::get_sum_cc_nodes() {
    int sum = 0;
    for (TreeLevel * l : levels ) {
        sum += l->num_cc_nodes;
    }
    return sum;
}

void SCC::TreeLevel::mark_for_first_round() {
    // mark the old parents' siblings
    // for each marked node in the first round
    // we grab the siblings of the parents of the node
    // and we grab the best neighbor too
    #ifdef TIME_SCC
    auto st_mark_first = utils::get_time();
    #endif
    std::set<SCC::TreeLevel::TreeNode *> to_add;
    for (SCC::TreeLevel::TreeNode * u_node : marked_nodes) {
        if (u_node->parent != NULL) {
            for (const auto & sib: u_node->parent->children) {
                SCC::TreeLevel::TreeNode * sib_node = sib.second;
                sib_node->marked_time = global_step;
                if (marked_node_set.find(sib_node) == marked_node_set.end()) {
                    to_add.insert(sib_node);
                    marked_node_set.insert(sib_node);
                }
            }
        }
    }
    for (SCC::TreeLevel::TreeNode * t: to_add) {
        marked_nodes.push_back(t);
    }
    #ifdef TIME_SCC
    auto en_mark_first = utils::get_time();
    std::cout << "#time mark_first " << utils::timedur(st_mark_first, en_mark_first) << std::endl; 
    #endif
}

void SCC::set_level_global_step() {
    for (size_t i=0; i < levels.size(); i++) {
        levels[i]->global_step = global_step;
    }
}

void SCC::TreeLevel::summary_message() {
    std::cout << "TreeLevel -";
    std::cout << " height=" << height;
    std::cout << " nodes=" << nodes.size();
    std::cout << " total_time=" << scc->total_time;
    std::cout << " best_neighbor_time=" << best_neighbor_time;
    std::cout << " cc_time=" << cc_time;
    std::cout << " graph_update_time=" << graph_update_time;
    std::cout << " overall_update_time=" << overall_update_time;
    std::cout << std::endl;
}

void SCC::print_structure() {

    std::cout << "SCC ------------------------------" << std::endl;
    for (size_t l=0; l < levels.size(); l++) {
        std::cout << "################################" << std::endl;
        std::cout << "LEVEL " << l << std::endl;
        std::cout << "Nodes: " << levels[l]->nodes.size() << " Marked: " << levels[l]->marked_nodes.size() << std::endl;
        for (size_t i=0; i < levels[l]->nodes.size(); i++) {
            std::cout << "nodes[ " << i << "].this_id " << levels[l]->nodes[i]->this_id << " .deleted " <<  levels[l]->nodes[i]->deleted << " .best_neighbor " <<  levels[l]->nodes[i]->best_neighbor << " .f "  << levels[l]->nodes[i]->f << " .last_parent " << levels[l]->nodes[i]->last_parent << " last updated " <<  levels[l]->nodes[i]->last_updated << " created time " <<  levels[l]->nodes[i]->created_time << " global " << global_step << " ccn ";
            for (SCC::TreeLevel::TreeNode * ccn : levels[l]->nodes[i]->cc_neighbors) {
                std::cout << " " << ccn ;
            }
            std::cout << " best_neighbors ";
            for (SCC::TreeLevel::TreeNode * ccn : levels[l]->nodes[i]->best_neighbors) {
                std::cout << " " << ccn ;
            }
            std::cout << " neighbors ";
            for (const auto & pair : levels[l]->nodes[i]->neigh) {
                std::cout << " (" << pair.first << ", " << pair.second << ")" ;
            }
            std::cout << std::endl;
        }
        std::cout << "################################" << std::endl;
    }
    std::cout << "SCC ------------------------------" << std::endl;

}