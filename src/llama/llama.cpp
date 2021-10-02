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

#include "llama.h"

/**
 * Construct an instance of a DAG structured clustering.
 * @param r Vector of indices
 * @param c Vector of indices
 * @param s s[i] is the similarity between r[i] and c[i]
 * @param linkage linkage function to use either integer (0 for single, 1 for average, 2 for approx. average).
 * @param num_rounds the number of rounds to use.
 * @param thresholds array (float32) of the minimum similarity to allow in an agglomeration 
 * @param cores number of parallel threads to use
 * @param max_num_parents maximum number of parents any node can have
 * @param max_num_neighbors maximum number of neigbhors any node can have in the graph 
 * @param lowest_value value used for missing / minimum similarity
 */
LLAMA *LLAMA::from_graph(
    std::vector<uint32_t> r,
    std::vector<uint32_t> c,
    std::vector<Eigen::VectorXf::Scalar> s,
    unsigned linkage,
    unsigned num_rounds,
    scalar *thresholds,
    unsigned cores,
    unsigned max_num_parents,
    unsigned max_num_neighbors,
    scalar lowest_value)
{
    LLAMA *dagclust = NULL;
    dagclust = new LLAMA(r, c, s, linkage, num_rounds, thresholds, cores, max_num_parents, max_num_neighbors, lowest_value);
    return dagclust;
}

/**
 * Run DAG structured clustering.
 */
void LLAMA::cluster()
{
    std::cout << "Starting llama clustering... " << std::endl;
    auto st_cluster = utils::get_time();
    unsigned i = 0;
    while (i < num_rounds)
    {
        auto st_round = utils::get_time();
        std::cout << "Starting round " << i << " with " << active_nodes.size() << " nodes." << std::endl;
        if (active_nodes.size() == 1) {
            break;
        }
        perform_round(thresholds[i]);
        auto en_round = utils::get_time();
        std::cout << "Ending round " << i << " with " << active_nodes.size() << " nodes in " << utils::timedur(st_round,en_round) << " seconds." << std::endl;
        i += 1;
    }
    auto en_cluster = utils::get_time();
    std::cout << "Ending llama clustering in " << utils::timedur(st_cluster, en_cluster) << " seconds." << std::endl;
    clustering_run = true;
}

void LLAMA::perform_round(scalar threshold)
{
    one_nn(threshold);
    propose_parents();
    contract();
    prune_to_k_neighbors();
    round_id += 1;
}

void LLAMA::one_nn(scalar threshold)
{
    for (LLAMANode * m_node : active_nodes) {
        scalar best_score = lowest_value;
        m_node->best_neighbor = m_node;
        m_node->best_neighbor_score = lowest_value;
        scalar m_count = (scalar) m_node->count;
        for (const auto &pair : m_node->neighbors)
        {
            if (pair.first != m_node)
            {
                scalar s = (pair.second / (m_count * (scalar)pair.first->count));
                if (s > best_score && s > threshold)
                {
                    best_score = s;
                    m_node->best_neighbor = pair.first;
                    m_node->best_neighbor_score = best_score;
                }
                else if (s == best_score && s > threshold && pair.first->ID < m_node->best_neighbor->ID)
                {
                    best_score = s;
                    m_node->best_neighbor = pair.first;
                    m_node->best_neighbor_score = best_score;
                }
            }
        }
    }
}

const bool LLAMA::parent_comp(const std::pair<LLAMANode *, scalar> &p1,
                              const std::pair<LLAMANode *, scalar> &p2)
{
    if (p1.second != p2.second)
    {
        return p1.second > p2.second;
    }
    else
    {
        return p1.first->ID < p2.first->ID;
    }
}

void LLAMA::contract()
{
    if (linkage == 0)
    {
        contract_single();
    }
    else if (linkage == 1)
    {
        contract_set_average();
    }
    else if (linkage == 2)
    {
        contract_bag_average();
    }
    else
    {
        throw "Undefined linkage! Allowable linkages Single = 0, Set Average = 1, Bag Average = 2";
    }
}

void LLAMA::contract_bag_average()
{
    auto c1_st = get_time();
    std::unordered_set<LLAMANode *> new_active_nodes;

    // Clear any old parents
    for (LLAMANode * m_node  : active_nodes) {
        m_node->parents.clear();
    }

    // add parents
    for (LLAMANode * m_node : active_nodes) {
        m_node->parents.emplace_back(m_node->chosen_parent, m_node->best_neighbor_score);
        // if we are not mutual nearest neighbors, then we need to add my parent as
        // my neighbors parent.
        if (m_node->best_neighbor->chosen_parent != m_node->chosen_parent)
        {
            m_node->best_neighbor->parents.emplace_back(m_node->chosen_parent, m_node->best_neighbor_score);
        }
    }

    // Filter parents
    // Each node only picks top K parents according to the score.
    if (max_num_parents > 0)
    {
        // std::cout << "Starting to prune parents..." << std::endl;
        // std::cout << "Pruning to " << max_num_parents << " parents..." << std::endl;
        // Set parent_counts to be zeros
        // // std::cout << "Zeroing parent_counts..." << std::endl;
        // auto prune_parents_st = get_time();
        for (LLAMANode * m_node: active_nodes) {
            m_node->parent_count = 0;
            m_node->new_node_count = 0;
        }

        // std::cout << "Zeroing parent_counts...Done!" << std::endl;

        // std::cout << "Sorting & restricting..." << std::endl;
        for (LLAMANode * m_node: active_nodes) {
            std::sort(m_node->parents.begin(), m_node->parents.end(), parent_comp);
            if (m_node->parents.size() > max_num_parents)
            {
                m_node->parents.resize(max_num_parents);
            }
            // count the number of times each parent is used
            for (const auto &p : m_node->parents)
            {
                p.first->parent_count += 1;
            }
        }
        // std::cout << "Sorting & restricting...Done!" << std::endl;

        // std::cout << "Selecting parents that are agreed upon..." << std::endl;
        for (LLAMANode * m_node : active_nodes) {
            // sort and restrict each nodes parents
            std::vector<std::pair<LLAMANode *, scalar>> tmp;
            for (const auto &p : m_node->parents)
            {
                if (p.first->parent_count == 2)
                {
                    tmp.emplace_back(p.first, p.second);
                }
            }
            m_node->parents.clear();

            if (tmp.empty())
            {
               m_node->parents.emplace_back(m_node, 0.0);
               m_node->skip = false;
               m_node->best_neighbor = m_node;
               m_node->chosen_parent = m_node;
               m_node->best_neighbor_score = 0.0;
            }
            else
            {
                for (const auto &t : tmp)
                {
                    m_node->parents.emplace_back(t.first, t.second);
                }
                bool found1 = false;
                for (const auto p : m_node->parents)
                {
                    if (p.first == m_node->chosen_parent)
                    {
                        found1 = true;
                    }
                }
                if (!found1)
                {
                   m_node->skip = false;
                   m_node->best_neighbor = m_node;
                   m_node->best_neighbor_score = 0.0;
                   m_node->parents.emplace_back(m_node, 0.0);
                   m_node->chosen_parent = m_node;
                }
            }
        }
        // std::cout << "Selecting parents that are agreed upon...Done!" << std::endl;

        // std::cout << "Selecting new_active_nodes..." << std::endl;
        for (LLAMANode * m_node: active_nodes)
        {
            for (const auto &p : m_node->parents)
            {
                // std::cout << "[new_active_nodes] m=" << m << " p.first=" << p.first << " all_nodes[m]->parents.size()=" << all_nodes[m]->parents.size() << " all_nodes[m]->best_neighbor_id=" << all_nodes[m]->best_neighbor_id << " all_nodes[m]->f=" << all_nodes[m]->f << " parent_counts[all_nodes[m]->f]=" << parent_counts[all_nodes[m]->f] << std::endl;
                new_active_nodes.insert(p.first);
            }
            m_node->parent_count = 0;
        }
        // std::cout << "Selecting new_active_nodes...Done!" << std::endl;
        // std::cout << "Selecting new_active_nodes...Done! new_active_nodes.size()=" << new_active_nodes.size() << std::endl;

        // auto prune_parents_end = get_time();
        // auto pp_time = sec(prune_parents_end, prune_parents_st);
        // std::cout << "Starting to prune parents...Done! in " << pp_time <<  " seconds." << std::endl;
    } else {
        for (LLAMANode * m_node: active_nodes) {
            new_active_nodes.insert(m_node->chosen_parent);
        }
    }

    for (LLAMANode * m_node: active_nodes) {
        // in the next round, we will include all but the nodes marked skipped
        if (!m_node->skip)
        {
            // we are always out-degree centric.
            // grab the counts from your neighbor
            if (m_node->best_neighbor != m_node)
            {
                m_node->new_node_count += m_node->best_neighbor->count;
            }
        }
    }
    for (LLAMANode * m_node : active_nodes) 
    {
        if (!m_node->skip)
        {
            m_node->count += m_node->new_node_count;
        }
    }

    save_parents(new_active_nodes);

    auto c1_end = get_time();
    auto c1_time = sec(c1_end, c1_st);
    // std::cout << "Contract Step1: " << c1_time << " seconds." << std::endl;

    // now we add graph edges for the next round
    auto c2_st = get_time();
    for (LLAMANode * m_node: active_nodes) {
        if (!m_node->skip)
        {
            // we will be updating the parent m chose.
            // update for m's neighbors
            for (const auto &pair : m_node->neighbors)
            {
                // counts go into
                for (const auto &parent : pair.first->parents)
                {
                    // std::cout << "m " << m << "m_node->best_neighbor_id " << m_node->best_neighbor_id << "neighbor " << pair.first << " neighbor score" << pair.second << " parent.first " << parent.first << " parent.second " << parent.second << " node_counts[m]" << node_counts[m] << " node_counts[parent.first]" << node_counts[parent.first] << std::endl;
                    // remove self loop
                    if (parent.first != m_node)
                    {
                        m_node->new_neighbors[parent.first] += pair.second;
                    }
                }
            }
            // std::cout << "m " << m << " m_node->best_neighbor_id " << m_node->best_neighbor_id << std::endl;
            // grab neighbors distances
            if (m_node->best_neighbor != m_node)
            {

                // std::cout << "m " << m << " != m_node->best_neighbor_id " << m_node->best_neighbor_id << std::endl;
                for (const auto &pair : m_node->best_neighbor->neighbors)
                {
                    // std::cout << "m " << m << " pair.first " << pair.first << " pair.second " << pair.second << std::endl;

                    // counts go into
                    for (const auto &parent : pair.first->parents)
                    {
                        // std::cout << " m " << m << "m_node->best_neighbor_id " << m_node->best_neighbor_id << "neighbor " << pair.first << " parent.first " << parent.first << " parent.second " << parent.second << std::endl;
                        bool parent_eq_neigh = parent.first == m_node->best_neighbor;

                        // std::cout << "m " << m << " parent.first " << parent.first << " parent.second " << parent.second << std::endl;
                        // remove self loop
                        if (parent.first != m_node)
                        {
                            m_node->new_neighbors[parent.first] += pair.second;
                        }
                    }
                }
            }
        }
    }

    for (LLAMANode * m_node : active_nodes)
    {
        if (!m_node->skip)
        {
            // std::cout << "m " << m << " m_node->skip " << m_node->skip << " m_node->new_neighbors.size() " << m_node->new_neighbors.size() << " m_node->neighbors.size() " << m_node->neighbors.size() << std::endl;
            m_node->neighbors.clear();
            for (const auto p : m_node->new_neighbors)
            {
                m_node->neighbors[p.first] = p.second;
            }
            // m_node->neighbors.insert(m_node->new_neighbors.begin(), m_node->new_neighbors.end());
            m_node->new_neighbors.clear();
            m_node->parents.clear();
        }
        else
        {
            m_node->neighbors.clear();
            m_node->parents.clear();
            m_node->new_neighbors.clear();
        }
    }
    // auto c2_end = get_time();
    // auto c2_time = sec(c2_st, c2_end);
    // std::cout << "Contract Step2: " << c2_time << " seconds." << std::endl;

    // auto c3_st = get_time();
    active_nodes.clear();
    for (LLAMANode * x : new_active_nodes)
    {
        active_nodes.push_back(x);
    }

    // auto c3_end = get_time();
    // auto c3_time = sec(c3_end, c3_st);

    // std::cout << "Contract Step3: " << c3_time << " seconds." << std::endl;
}

void LLAMA::contract_set_average()
{
    // each node needs to do the following:
    auto c1_st = get_time();

    if (round_id == 0)
    {
        // std::cout << "First round... setting up descendants..." << std::endl;
        for (LLAMANode * m_node : active_nodes) {
            m_node->descendants.insert(m_node->ID);
            for (const auto &p : m_node->neighbors)
            {
                m_node->leafs.insert(p.first->ID);
                m_node->leaf_sims[p.first->ID] = p.second;
            }
        }
        // std::cout << "First round... setting up descendants...Done!" << std::endl;
    }

    std::unordered_set<LLAMANode *> new_active_nodes;

    // Clear any old parents
    for (const auto &m_node : active_nodes)
    {
        m_node->parents.clear();
    }

    for (const auto &m_node : active_nodes) {
        m_node->parents.emplace_back(m_node->chosen_parent, m_node->best_neighbor_score);
        // need to be careful about depulicates is mutual nn's
        // if my neighbor has a different parent than me
        // assign my neighbor me as a parent
        if (m_node->best_neighbor->chosen_parent != m_node->chosen_parent)
        {
            m_node->best_neighbor->parents.emplace_back(m_node->chosen_parent, m_node->best_neighbor_score);
        }
    }

    // Filter parents
    // Each node only picks top K parents according to the score.
    if (max_num_parents > 0)
    {
        // std::cout << "Starting to prune parents..." << std::endl;
        // std::cout << "Pruning to " << max_num_parents << " parents..." << std::endl;
        auto prune_parents_st = get_time();
 
        // Set parent_counts to be zeros
        // std::cout << "Zeroing parent_counts..." << std::endl;
        for (const auto &m_node : active_nodes) {
            m_node->parent_count = 0;
            m_node->new_node_count = 0;
        }
        // std::cout << "Zeroing parent_counts...Done!" << std::endl;

        // std::cout << "Sorting & restricting..." << std::endl;
        for (const auto &m_node : active_nodes) {
            std::sort(m_node->parents.begin(), m_node->parents.end(), parent_comp);
            if (m_node->parents.size() > max_num_parents)
            {
                m_node->parents.resize(max_num_parents);
            }
            // count the number of times each parent is used
            for (const auto &p : m_node->parents)
            {
                p.first->parent_count += 1;
            }
        }
        // std::cout << "Sorting & restricting...Done!" << std::endl;

        // std::cout << "Selecting parents that are agreed upon..." << std::endl;
        for (const auto &m_node : active_nodes) {
            // sort and restrict each nodes parents
            std::vector<std::pair<LLAMANode *, scalar>> tmp;
            for (const auto &p : m_node->parents)
            {
                if (p.first->parent_count == 2)
                {
                    tmp.emplace_back(p.first, p.second);
                }
            }

            m_node->parents.clear();
            if (tmp.empty())
            {
                m_node->parents.emplace_back(m_node, 0.0);
                m_node->skip = false;
                m_node->best_neighbor = m_node;
                m_node->chosen_parent = m_node;
                m_node->best_neighbor_score = 0.0;
            }
            else
            {
                for (const auto &t : tmp)
                {
                    m_node->parents.emplace_back(t.first, t.second);
                }
                bool found1 = false;
                for (const auto p : m_node->parents)
                {
                    if (p.first == m_node->chosen_parent)
                    {
                        found1 = true;
                    }
                }
                if (!found1)
                {
                    // std::cout << "[not found1] m=" << m << " m_node->best_neighbor_id=" << m_node->best_neighbor_id << " m_node->f=" << m_node->f << " parent_counts[m_node->f]=" << parent_counts[m_node->f] << std::endl;
                    m_node->skip = false;
                    m_node->best_neighbor = m_node;
                    m_node->best_neighbor_score = 0.0;
                    m_node->parents.emplace_back(m_node, 0.0);
                    m_node->chosen_parent = m_node;
                }
            }
        }
        // std::cout << "Selecting parents that are agreed upon...Done!" << std::endl;

        // std::cout << "Selecting new_active_nodes..." << std::endl;
        for (const auto &m_node : active_nodes)
        {
            for (const auto &p : m_node->parents)
            {
                // std::cout << "[new_active_nodes] m=" << m << " p.first=" << p.first << " all_nodes[m].parents.size()=" << all_nodes[m].parents.size() << " all_nodes[m].best_neighbor_id=" << all_nodes[m].best_neighbor_id << " all_nodes[m].f=" << all_nodes[m].f << " parent_counts[all_nodes[m].f]=" << parent_counts[all_nodes[m].f] << std::endl;
                new_active_nodes.insert(p.first);
            }
            m_node->parent_count = 0;
        }
        // std::cout << "Selecting new_active_nodes...Done!" << std::endl;
        // std::cout << "Selecting new_active_nodes...Done! new_active_nodes.size()=" << new_active_nodes.size() << std::endl;

        auto prune_parents_end = get_time();
        auto pp_time = sec(prune_parents_end, prune_parents_st);
        // std::cout << "Starting to prune parents...Done! in " << pp_time <<  " seconds." << std::endl;
    }

    save_parents(new_active_nodes);

    auto c1_end = get_time();
    auto c1_time = sec(c1_end, c1_st);
    // std::cout << "Contract Step1: " << c1_time << " seconds." << std::endl;

    // collect my new descendants
    // std::cout << "Collecting descendants..." << std::endl;

    for (LLAMANode * m_node : active_nodes)
    {
        if (!m_node->skip)
        {
            if (m_node->best_neighbor != m_node)
            {
                m_node->new_descendants.insert(m_node->best_neighbor->descendants.begin(),
                                                    m_node->best_neighbor->descendants.end());
            }
        }
    }
    // std::cout << "Collecting descendants...Done!" << std::endl;
    // std::cout << "Update descendants..." << std::endl;
    // update my descendants
    for (LLAMANode * m_node : active_nodes)  {
        if (!m_node->skip)
        {
            if (m_node->best_neighbor != m_node)
            {
                m_node->descendants.insert(m_node->new_descendants.begin(), m_node->new_descendants.end());
                m_node->new_descendants.clear();
            }
        }
        else
        {
            m_node->new_descendants.clear();
        }
    }
    // std::cout << "Update descendants...Done!" << std::endl;

    // now we add graph edges for the next round
    // std::cout << "Computing distances..." << std::endl;

    auto c2_st = get_time();
    for (LLAMANode * m_node : active_nodes)  {
        if (!m_node->skip)
        {
            // we will be updating the parent m chose.
            // update for m's neighbors
            for (const auto &pair : m_node->neighbors)
            {
                // counts go into
                for (const auto &parent : pair.first->parents)
                {
                    // remove self loop
                    if (parent.first != m_node)
                    {
                        if (m_node->new_neighbors.find(parent.first) == m_node->new_neighbors.end())
                        {
                            m_node->new_neighbors[parent.first] = set_avg(m_node, parent.first);
                        }
                    }
                }
            }
            // grab neighbors distances
            if (m_node->best_neighbor != m_node)
            {
                for (const auto &pair : m_node->best_neighbor->neighbors)
                {
                    // counts go into
                    for (const auto &parent : pair.first->parents)
                    {
                        if (parent.first != m_node)
                        {
                            if (m_node->new_neighbors.find(parent.first) == m_node->new_neighbors.end())
                            {
                                m_node->new_neighbors[parent.first] = set_avg(m_node, parent.first);
                            }
                        }
                    }
                }
            }
        }
    }
    // std::cout << "Computing distances...Done!" << std::endl;

    for (LLAMANode * m_node : active_nodes)  {
        if (!m_node->skip)
        {
            m_node->neighbors.clear();
            for (const auto p : m_node->new_neighbors)
            {
                m_node->neighbors[p.first] = p.second;
            }
            m_node->new_neighbors.clear();
            m_node->parents.clear();
        }
        else
        {
            m_node->neighbors.clear();
            m_node->parents.clear();
            m_node->new_neighbors.clear();
        }
    }
    auto c2_end = get_time();
    auto c2_time = sec(c2_st, c2_end);
    // std::cout << "Contract Step2: " << c2_time << " seconds." << std::endl;

    auto c3_st = get_time();
    active_nodes.clear();
    for (LLAMANode * x : new_active_nodes)
    {
        active_nodes.push_back(x);
    }

    auto c3_end = get_time();
    auto c3_time = sec(c3_end, c3_st);

    // std::cout << "Contract Step3: " << c3_time << " seconds." << std::endl;
}

void LLAMA::contract_single()
{
    // each node needs to do the following:
    auto c1_st = get_time();
    std::unordered_set<LLAMANode *> new_active_nodes;

    // Clear any old parents
    for (LLAMANode * m_node : active_nodes)
    {
        m_node->parents.clear();
    }
    for (LLAMANode * m_node : active_nodes) {
        m_node->parents.emplace_back(m_node->chosen_parent, m_node->best_neighbor_score);
        // need to be careful about depulicates is mutual nn's
        // if my neighbor has a different parent than me
        // assign my neighbor me as a parent
        if (m_node->best_neighbor->chosen_parent != m_node->chosen_parent)
        {
            m_node->best_neighbor->parents.emplace_back(m_node->chosen_parent, m_node->best_neighbor_score);
        }
    }

    // Filter parents
    // Each node only picks top K parents according to the score.
    if (max_num_parents > 0)
    {
        // std::cout << "Starting to prune parents..." << std::endl;
        // std::cout << "Pruning to " << max_num_parents << " parents..." << std::endl;
        auto prune_parents_st = get_time();
        new_active_nodes.clear();

        // Set parent_counts to be zeros
        // std::cout << "Zeroing parent_counts..." << std::endl;
        for (LLAMANode * m_node : active_nodes) {
            m_node->parent_count = 0;
            m_node->new_node_count = 0;
        }
        // std::cout << "Zeroing parent_counts...Done!" << std::endl;

        // std::cout << "Sorting & restricting..." << std::endl;
        for (LLAMANode * m_node : active_nodes) {
            // if (all_nodes[m].parents.size() == 0) {
            //     std::cerr << "all_nodes[m].parents.size()=" << all_nodes[m].parents.size() << " m=" << m << std::endl;
            // }
            std::sort(m_node->parents.begin(), m_node->parents.end(), parent_comp);
            if (m_node->parents.size() > max_num_parents)
            {
                m_node->parents.resize(max_num_parents);
            }
            // count the number of times each parent is used
            for (const auto &p : m_node->parents)
            {
                p.first->parent_count += 1;
            }
        }
        // std::cout << "Sorting & restricting...Done!" << std::endl;

        // std::cout << "Selecting parents that are agreed upon..." << std::endl;
        for (LLAMANode * m_node : active_nodes) {
            // sort and restrict each nodes parents
            std::vector<std::pair<LLAMANode *, scalar>> tmp;
            for (const auto &p : m_node->parents)
            {
                if (p.first->parent_count == 2)
                {
                    tmp.emplace_back(p.first, p.second);
                }
            }

            m_node->parents.clear();
            if (tmp.empty())
            {
                m_node->parents.emplace_back(m_node, 0.0);
                m_node->skip = false;
                m_node->best_neighbor = m_node;
                m_node->chosen_parent = m_node;
                m_node->best_neighbor_score = 0.0;
            }
            else
            {
                for (const auto &t : tmp)
                {
                    m_node->parents.emplace_back(t.first, t.second);
                }
                bool found1 = false;
                for (const auto p : m_node->parents)
                {
                    if (p.first == m_node->chosen_parent)
                    {
                        found1 = true;
                    }
                }
                if (!found1)
                {
                    // std::cout << "[not found1] m=" << m << " all_nodes[m].best_neighbor_id=" << all_nodes[m].best_neighbor_id << " all_nodes[m].f=" << all_nodes[m].f << " parent_counts[all_nodes[m].f]=" << parent_counts[all_nodes[m].f] << std::endl;
                    m_node->skip = false;
                    m_node->best_neighbor = m_node;
                    m_node->best_neighbor_score = 0.0;
                    m_node->parents.emplace_back(m_node, 0.0);
                    m_node->chosen_parent = m_node;
                }
            }
        }
        // std::cout << "Selecting parents that are agreed upon...Done!" << std::endl;

        // std::cout << "Selecting new_active_ids..." << std::endl;
        for (LLAMANode * m_node : active_nodes) {
            for (const auto &p : m_node->parents)
            {
                // std::cout << "[new_active_ids] m=" << m << " p.first=" << p.first << " all_nodes[m].parents.size()=" << all_nodes[m].parents.size() << " all_nodes[m].best_neighbor_id=" << all_nodes[m].best_neighbor_id << " all_nodes[m].f=" << all_nodes[m].f << " parent_counts[all_nodes[m].f]=" << parent_counts[all_nodes[m].f] << std::endl;
                new_active_nodes.insert(p.first);
            }
            m_node->parent_count = 0;
        }
        // std::cout << "Selecting new_active_ids...Done!" << std::endl;
        // std::cout << "Selecting new_active_ids...Done! new_active_ids.size()=" << new_active_ids.size() << std::endl;

        auto prune_parents_end = get_time();
        auto pp_time = sec(prune_parents_end, prune_parents_st);
        // std::cout << "Starting to prune parents...Done! in " << pp_time <<  " seconds." << std::endl;
    }

    save_parents(new_active_nodes);

    auto c1_end = get_time();
    auto c1_time = sec(c1_end, c1_st);
    // std::cout << "Contract Step1: " << c1_time << " seconds." << std::endl;

    // now we add graph edges for the next round
    auto c2_st = get_time();
    for (LLAMANode * m_node : active_nodes) {
        if (!m_node->skip)
        {
            // we will be updating the parent m chose.
            // update for m's neighbors
            for (const auto &pair : m_node->neighbors)
            {

                // counts go into
                for (const auto &parent : pair.first->parents)
                {
                    // std::cout << "m " << m << "all_nodes[m].best_neighbor_id " << all_nodes[m].best_neighbor_id << "neighbor " << pair.first << " neighbor score" << pair.second << " parent.first " << parent.first << " parent.second " << parent.second << " node_counts[m]" << node_counts[m] << " node_counts[parent.first]" << node_counts[parent.first] << std::endl;
                    // this is a tricky case
                    // if my neighbor is a singleton node
                    // and it was the thing i merged with
                    // i can't merge with it again.
                    bool parent_eq_neigh = parent.first == m_node->best_neighbor;

                    // remove self loop
                    if (parent.first != m_node)
                    {
                        if (m_node->new_neighbors.find(parent.first) == m_node->new_neighbors.end())
                        {
                            m_node->new_neighbors[parent.first] = pair.second;
                        }
                        else if (m_node->new_neighbors[parent.first] < pair.second)
                        {
                            m_node->new_neighbors[parent.first] = pair.second;
                        }
                    }
                }
            }
            // std::cout << "m " << m << " m_node->best_neighbor_id " << m_node->best_neighbor_id << std::endl;
            // grab neighbors distances
            if (m_node->best_neighbor != m_node)
            {

                // std::cout << "m " << m << " != all_nodes[m].best_neighbor_id " << all_nodes[m].best_neighbor_id << std::endl;
                for (const auto &pair : m_node->best_neighbor->neighbors)
                {
                    // std::cout << "m " << m << " pair.first " << pair.first << " pair.second " << pair.second << std::endl;

                    // counts go into
                    for (const auto &parent : pair.first->parents)
                    {
                        // std::cout << " m " << m << "all_nodes[m].best_neighbor_id " << all_nodes[m].best_neighbor_id << "neighbor " << pair.first << " parent.first " << parent.first << " parent.second " << parent.second << std::endl;
                        bool parent_eq_neigh = parent.first == m_node->best_neighbor;

                        // std::cout << "m " << m << " parent.first " << parent.first << " parent.second " << parent.second << std::endl;
                        // remove self loop
                        if (m_node->new_neighbors.find(parent.first) == m_node->new_neighbors.end())
                        {
                            m_node->new_neighbors[parent.first] = pair.second;
                        }
                        else if (m_node->new_neighbors[parent.first] < pair.second)
                        {
                            m_node->new_neighbors[parent.first] = pair.second;
                        }
                    }
                }
            }
        }
    }

    for (LLAMANode * m_node : active_nodes) {
        if (!m_node->skip)
        {
            // std::cout << "m " << m << " all_nodes[m].skip " << all_nodes[m].skip << " all_nodes[m].new_neighbors.size() " << all_nodes[m].new_neighbors.size() << " all_nodes[m].neighbors.size() " << all_nodes[m].neighbors.size() << std::endl;
            m_node->neighbors.clear();
            for (const auto p : m_node->new_neighbors)
            {
                m_node->neighbors[p.first] = p.second;
            }
            // all_nodes[m].neighbors.insert(all_nodes[m].new_neighbors.begin(), all_nodes[m].new_neighbors.end());
            m_node->new_neighbors.clear();
            m_node->parents.clear();
        }
        else
        {
            m_node->neighbors.clear();
            m_node->parents.clear();
            m_node->new_neighbors.clear();
        }
    }
    auto c2_end = get_time();
    auto c2_time = sec(c2_st, c2_end);
    // std::cout << "Contract Step2: " << c2_time << " seconds." << std::endl;

    auto c3_st = get_time();
    active_nodes.clear();
    for (LLAMANode * x : new_active_nodes)
    {
        active_nodes.push_back(x);
    }

    auto c3_end = get_time();
    auto c3_time = sec(c3_end, c3_st);

    // std::cout << "Contract Step3: " << c3_time << " seconds." << std::endl;
}

scalar LLAMA::set_avg(LLAMANode * a, LLAMANode * b)
{
    scalar s = 0.0;
    for (const auto da : a->descendants)
    {
        std::vector<node_id_t> dbi;
        std::set_intersection(all_nodes[da]->leafs.begin(),
                              all_nodes[da]->leafs.end(),
                              b->descendants.begin(),
                              b->descendants.end(), std::back_inserter(dbi));
        for (const auto db : dbi)
        {
            s += all_nodes[da]->leaf_sims[db];
        }
    }
    return s / ((scalar)a->descendants.size() * (scalar)b->descendants.size());
}

void LLAMA::get_child_parent_edges()
{
    if (clustering_run && children.empty()) 
    {
        children.clear();
        parents.clear();
        auto build_desc_st = get_time();
        // std::cout << "number of active ids: " << number_of_active_ids.size() << std::endl;
        int offset = 0;
        for (int r = 0; r < number_of_active_ids.size(); r++)
        {
            for (int m = 0; m < number_of_active_ids[r]; m++)
            {
                for (const auto c : all_parent2children[r][m])
                {
                    children.push_back(c + offset);
                    parents.push_back(m + offset + number_of_active_ids[r]);
                }
            }
            offset += number_of_active_ids[r];
            // std::cout << "children.size() " << children.size() << std::endl;
            // std::cout << "parents.size() " << parents.size() << std::endl;
        }
        auto build_desc_end = get_time();
        auto build_desc_time = sec(build_desc_st, build_desc_end);
    }
}

void LLAMA::set_descendants()
{
    if (clustering_run && all_node2descendants_len == 0) 
    {
        // std::cout << "Building descendants... " << std::endl;
        size_t num_rounds_so_far = number_of_active_ids.size();
        // std::cout << "Number of rounds: " << num_rounds_so_far << std::endl;

        auto build_desc_st = get_time();
        std::unordered_set<node_id_t> *last_round_descandants;

        assert (all_node2descendants_len == 0);

        all_node2descendants = new std::unordered_set<node_id_t> *[num_rounds_so_far];
        bool *all_node2pruned[num_rounds_so_far];
        all_node2descendants_len = num_rounds_so_far;
        // for each round
        for (int r = 0; r < number_of_active_ids.size(); r++)
        {
            // std::cout << "Working on round " << r << std::endl;
            all_node2descendants[r] = new std::unordered_set<node_id_t>[number_of_active_ids[r]];
            all_node2pruned[r] = new bool[number_of_active_ids[r]];
            // for each parent node in the round
            // merge descendant maps of the children
            utils::parallel_for(cores, 0, number_of_active_ids[r], [&](node_id_t m) -> void
                                            {
                                                if (r == 0)
                                                {
                                                    all_node2descendants[r][m].insert(all_active_ids[r][m]);
                                                    all_node2pruned[r][m] = false;
                                                }
                                                else
                                                {
                                                    for (const auto c : all_parent2children[r][m])
                                                    {
                                                        // prune only children
                                                        all_node2pruned[r - 1][c] = all_parent2children[r][m].size() == 1;
                                                        all_node2descendants[r][m].insert(all_node2descendants[r - 1][c].begin(),
                                                                                        all_node2descendants[r - 1][c].end());
                                                    }
                                                }
                                            });
        }
        auto build_desc_end = get_time();
        auto build_desc_time = sec(build_desc_st, build_desc_end);
        // std::cout << "Building descendants...: " << build_desc_time << " seconds." << std::endl;

        // std::cout << "Set descendants_c...: " << std::endl;
        auto build_desc_start1 = get_time();
        // rounds
        int offset = 0;
        std::map<node_id_t, node_id_t> uniqMap;
        for (int r = 0; r < number_of_active_ids.size(); r++)
        {
            // std::cout << "Working on round " << r << std::endl;
            // nodes in round r
            for (int m = 0; m < number_of_active_ids[r]; m++)
            {
                // for each descendant of m in r
                if (!all_node2pruned[r][m])
                {
                    for (const auto &d : all_node2descendants[r][m])
                    {
                        descendants_r.push_back(d);
                        if (uniqMap.find(m + offset) == uniqMap.end())
                        {
                            uniqMap[m + offset] = uniqMap.size();
                        }
                        descendants_c.push_back(uniqMap[m + offset]);
                    }
                }
            }
            // delete[] all_node2pruned[r];
            offset += number_of_active_ids[r];
        }
        // delete[] all_node2pruned;
        auto build_desc_end1 = get_time();
        auto build_desc_time1 = sec(build_desc_start1, build_desc_end1);
    }
}

void LLAMA::save_parents(std::unordered_set<LLAMANode *> &parent_ids)
{
    auto sp_start = get_time();
    // if this is the first round
    if (round_id == 0)
    {
        // number of ids
        number_of_active_ids.push_back(active_nodes.size());
        // save the ids themselves
        node_id_t *this_round_ids;
        this_round_ids = new node_id_t[active_nodes.size()];
        std::unordered_map<node_id_t, node_id_t> parentid2idx;
        for (int i = 0; i < active_nodes.size(); i++)
        {
            this_round_ids[i] = active_nodes[i]->ID;
            parentid2idx[active_nodes[i]->ID] = i;
        }
        all_active_ids.push_back(this_round_ids);
        all_map_active_ids_to_seq_id.push_back(parentid2idx);
        // save parent 2 child map
        std::vector<node_id_t> *this_round_children;
        this_round_children = new std::vector<node_id_t>[active_nodes.size()];
        for (int i = 0; i < active_nodes.size(); i++)
        {
            this_round_children[parentid2idx[active_nodes[i]->ID]].push_back(parentid2idx[active_nodes[i]->ID]);
        }
        all_parent2children.push_back(this_round_children);
    }
    // now save parent to children
    number_of_active_ids.push_back(parent_ids.size());
    node_id_t *this_round_ids;
    this_round_ids = new node_id_t[parent_ids.size()];
    std::unordered_map<node_id_t, node_id_t> parentid2idx;
    int i = 0;
    for  (LLAMANode * n : parent_ids)
    {
        this_round_ids[i] = n->ID;
        parentid2idx[n->ID] = i;
        i++;
    }
    std::vector<node_id_t> *this_round_children;
    this_round_children = new std::vector<node_id_t>[parent_ids.size()];
    i = 0;
    for  (LLAMANode * n : active_nodes)
    {
        for (const auto &p : all_nodes[n->ID]->parents)
        {
            // map children to be sequential ids
            this_round_children[parentid2idx[p.first->ID]].push_back(all_map_active_ids_to_seq_id[round_id][n->ID]);
        }
    }
    all_parent2children.push_back(this_round_children);
    all_map_active_ids_to_seq_id.push_back(parentid2idx);
    auto sp_end = get_time();
    auto sp_time = sec(sp_end, sp_start);
}

void LLAMA::propose_parents()
{
    // std::cout << "checking mutual nn..." << std::endl;
    for (LLAMANode * m_node : active_nodes)  {
        if (m_node->best_neighbor->best_neighbor == m_node)
        {
            if (m_node->ID <= m_node->best_neighbor->ID)
            {
                m_node->chosen_parent = m_node;
                m_node->skip = false;
            }
            else
            {
                m_node->chosen_parent = m_node->best_neighbor;
                m_node->skip = true;
            }
        }
        else
        {
            m_node->chosen_parent = m_node;
            m_node->skip = false;
        }
    }
}

LLAMA::LLAMA(
    std::vector<uint32_t> r,
    std::vector<uint32_t> c,
    std::vector<Eigen::VectorXf::Scalar> s,
    unsigned linkage,
    unsigned num_rounds,
    scalar *thresholds,
    unsigned cores,
    unsigned max_num_parents,
    unsigned max_num_neighbors,
    scalar lowest_value)
{
    std::cout << "graphgrove - LLAMA Constructor....V0.0.2" << std::endl;
    std::cout << "num rows .... " << r.size() << std::endl;
    std::cout << "num cols .... " << c.size() << std::endl;
    std::cout << "num sims .... " << s.size() << std::endl;
    std::cout << "MAX_NODES .... " << MAX_NODES << std::endl;

    std::cout << "parameters .... " << std::endl;
    this->num_rounds = num_rounds;
    this->thresholds = thresholds;
    this->cores = cores;
    this->max_num_parents = max_num_parents;
    this->max_num_neighbors = max_num_neighbors;
    this->lowest_value = lowest_value;
    this->linkage = linkage;
    std::cout << "num_rounds .... " << this->num_rounds << std::endl;
    std::cout << "cores .... " << this->cores << std::endl;
    std::cout << "linkage .... " << this->linkage << std::endl;
    std::cout << "num_rounds .... " << this->num_rounds << std::endl;
    std::cout << "max_num_parents .... " << this->max_num_parents << std::endl;
    std::cout << "max_num_neighbors .... " << this->max_num_neighbors << std::endl;
    std::cout << "lowest_value .... " << this->lowest_value << std::endl;
    std::cout << "building edge graph...." << std::endl;

    init_all_nodes();

    std::unordered_set<LLAMANode *> uniq_nodes;

    for (int i = 0; i < r.size(); i++)
    {
        utils::progressbar(i, r.size());
        LLAMANode * r_node = all_nodes[r[i]];
        LLAMANode * c_node = all_nodes[c[i]];
        if (r_node != c_node)
        {
            // std::cout << r[i] << " -> " << c[i] << std::endl;
            r_node->neighbors[c_node] = s[i];
            c_node->neighbors[r_node] = s[i];
        }
        r_node->count = 1; 
        c_node->count = 1;
        uniq_nodes.insert(r_node);
        uniq_nodes.insert(c_node);
    }
    for (const auto &x : uniq_nodes)
    {
        active_nodes.push_back(x);
    }
    num_points = active_nodes.size();
    std::cout << "num_points .... " << this->num_points << std::endl;
    std::cout << "building edge graph....Done!" << std::endl;
}

void LLAMA::prune_to_k_neighbors()
{
    if (max_num_neighbors > 0)
    {
        // std::cout << date_get_time() << "Running prune_to_k_neighbors..." << std::endl;
        auto st = get_time();
        for (LLAMANode * m_node: active_nodes) {
            std::priority_queue<std::pair<scalar, LLAMANode *>> pq;
            for (const auto &p :   m_node->neighbors)
            {
                pq.emplace(p.second, p.first);
            }
            m_node->neighbors.clear();
            size_t idx = 0;
            while (pq.size() > 0 && idx < max_num_neighbors)
            {
                m_node->neighbors[pq.top().second] = pq.top().first;
                pq.pop();
                idx += 1;
            }
        }
        // symmetrize
        for (LLAMANode * m_node: active_nodes) {
            for (const auto &p :  m_node->neighbors)
            {
                if (p.first->neighbors.find(m_node) != p.first->neighbors.end())
                {
                    p.first->neighbors[m_node] = p.second;
                }
            }
        }
        auto en = get_time();
        auto prune_time = sec(st, en);
        // std::cout << "Running prune_to_k_neighbors...Done in " << prune_time << " seconds." << std::endl;
    }
}