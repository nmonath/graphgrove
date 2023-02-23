/*
 * Copyright (c) 2021 The authors of SG Tree All rights reserved.
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


#include "sg_tree.h"
#include <iostream>

scalar* SGTree::compute_pow_table()
{
    scalar* powdict = new scalar[2048];
    for (int i = 0; i < 2048; ++i)
        powdict[i] = (scalar) pow(SGTree::base, i - 1024);
    return powdict;
}

scalar* SGTree::powdict = compute_pow_table();

#ifdef PRINTVER
std::map<int, std::atomic<unsigned>> SGTree::Node::dist_count;
#endif

/******************************* Insert ***********************************************/
bool SGTree::insert(SGTree::Node* current, const pointType& p, unsigned UID, scalar dist_current)
{
    bool result = false;
#ifdef DEBUG
    if (current->dist(p) > current->covdist())
        throw std::runtime_error("Internal insert got wrong input!");
    if (truncateLevel > 0 && current->level < maxScale - truncateLevel)
    {
        std::cout << maxScale;
        std::cout << " skipped" << std::endl;
        return false;
    }
#endif
    if (truncate_level > 0 && current->level < max_scale-truncate_level)
        return true;

    //acquire read lock
    current->mut.lock_shared();

    // Find the closest children
    unsigned num_children = unsigned(current->children.size());
    scalar dist_child = std::numeric_limits<scalar>::max();
    int child_idx = -1;
    for (unsigned i = 0; i < num_children; ++i)
    {
        scalar temp_dist = std::numeric_limits<scalar>::max();
        // don't recompute the distance in the case that nesting 
        // already exists in the tree structure. 
        if (current->children[i]->UID != current->UID)
        {
            temp_dist = current->children[i]->dist(p);
        } 
        else 
        {
            temp_dist = dist_current;
        }
        if (temp_dist < dist_child)
        {
            dist_child = temp_dist;
            child_idx = i;
        }
    }

    if (dist_child <= 0.0)
    {
        //release read lock then enter child
        current->mut.unlock_shared();
        std::cout << "Duplicate entry!!!" << std::endl;
        // std::cout << current->children[child_idx]->_p << std::endl;
        // std::cout << p << std::endl;
    }
    else if (use_nesting && dist_child > dist_current && dist_current <= current->sepdist()) 
    {
        // nesting case where we need to add a new copy of the current node as a child of itself.
        assert(current->UID != current->children[child_idx]->UID);
        //release read lock then acquire write lock
        current->mut.unlock_shared();
        current->mut.lock();
        // check if insert is still valid, i.e. no other point was inserted else restart
        if (num_children==current->children.size())
        {
            // create a new child, a copy of the current node (with new id, but same UID)
            int new_id = N++;
            Node * new_child = current->setChild(current->_p, current->UID, new_id);
            result = true;
            current->mut.unlock();

            int local_min = min_scale.load();
            while( local_min > current->level - 1){
                min_scale.compare_exchange_weak(local_min, current->level - 1, std::memory_order_relaxed, std::memory_order_relaxed);
                local_min = min_scale.load();
            }

            // now insert the new point into the new child.
            insert(new_child, p, UID, dist_current);
        }
        else
        {
            current->mut.unlock();
            result = insert(current, p, UID, dist_current);
        }

    }
    else if (dist_child <= current->sepdist() && (!use_nesting || dist_child <= dist_current))
    {
        //release read lock then enter child
        Node* child = current->children[child_idx];
        if (child->maxdistUB < dist_child)
           child->maxdistUB = dist_child;
        current->mut.unlock_shared();
        result = insert(child, p, UID, dist_child);
    }
    else
    {
        //release read lock then acquire write lock
        current->mut.unlock_shared();
        current->mut.lock();
        // check if insert is still valid, i.e. no other point was inserted else restart
        if (num_children==current->children.size())
        {
            int new_id = N++;
            current->setChild(p, UID, new_id);
            result = true;
            current->mut.unlock();

            int local_min = min_scale.load();
            while( local_min > current->level - 1){
                min_scale.compare_exchange_weak(local_min, current->level - 1, std::memory_order_relaxed, std::memory_order_relaxed);
                local_min = min_scale.load();
            }
        }
        else
        {
            current->mut.unlock();
            result = insert(current, p, UID, dist_current);
        }
    }
    return result;
}

void SGTree::calc_maxdist()
{
    std::vector<SGTree::Node*> travel;
    std::vector<SGTree::Node*> active;

    SGTree::Node* current = root;

    root->maxdistUB = 0.0;
    travel.push_back(root);
    while (travel.size() > 0)
    {
        current = travel.back();
        if (current->maxdistUB <= 0) {
            while (current->children.size() > 0)
            {
                active.push_back(current);
                // push the children
                for (int i = int(current->children.size()) - 1; i >= 0; --i)
                {
                    current->children[i]->maxdistUB = 0.0;
                    travel.push_back(current->children[i]);
                }
                current = current->children[0];
            }
        }
        else
            active.pop_back();

        // find distance with current node
        for (const auto& n : active)
            n->maxdistUB = std::max(n->maxdistUB, n->dist(current));

        // Pop
        travel.pop_back();
    }
}

bool SGTree::insert(const pointType& p, unsigned UID)
{
    bool result = false;
    id_valid = false;
    global_mut.lock_shared();
    scalar curr_root_dist = root->dist(p);
    if (curr_root_dist <= 0.0)
    {
        std::cout << "Duplicate entry!!!" << std::endl;
        // std::cout << root->_p << std::endl;
        // std::cout << p << std::endl;
    }
    else if (curr_root_dist > root->covdist())
    {
        std::cout<<"Entered case 1: " << root->dist(p) << " " << root->covdist() << " " << root->level <<std::endl;
        std::pair<SGTree::Node*, scalar> fn = FurthestNeighbour(p);
        global_mut.unlock_shared();
        std::cout<<"Requesting global lock!" <<std::endl;
        global_mut.lock();
        while (root->dist(p) > base * root->covdist()/(base-1))
        {
            SGTree::Node* current = root;
            SGTree::Node* parent = NULL;
            while (current->children.size() > 0)
            {
                parent = current;
                current = current->children.back();
            }
            if (parent != NULL)
            {
                parent->children.pop_back();
                std::pair<SGTree::Node*, scalar> fni = FurthestNeighbour(current->_p);
                current->level = root->level + 1;
                current->maxdistUB = fni.second;
                current->children.push_back(root);
                root = current;
            }
            else
            {
                root->level += 1;
            }
        }
        SGTree::Node* temp = new SGTree::Node;
        temp->_p = p;
        temp->level = root->level + 1;
        temp->ID = N++;
        temp->UID = UID;
        temp->maxdistUB = fn.second;
        temp->children.push_back(root);
        root = temp;
        max_scale = root->level;
        result = true;
        global_mut.unlock();
        global_mut.lock_shared();
    }
    else
    {
        // std::cout << "insert normal " << std::endl;
        result = insert(root, p, UID, curr_root_dist);
        // std::cout << "insert beam " << std::endl;
    }
    global_mut.unlock_shared();
    return result;
}

/******************************* Remove ***********************************************/

//TODO: Amortized implementation is needed


/****************************** Nearest Neighbour *************************************/

std::pair<SGTree::Node*, scalar> SGTree::NearestNeighbour(const pointType &p) const
{
    std::pair<SGTree::Node*, scalar> nn(root, root->dist(p));
    std::vector<std::pair<SGTree::Node*, scalar>> travel;
    SGTree::Node* curNode;
    scalar curDist;

    // Scratch memory
    std::vector<int> local_idx;
    std::vector<scalar> local_dists;
    auto comp_x = [&local_dists](int a, int b) { return local_dists[a] > local_dists[b]; };

    // Initialize with root
    travel.push_back(nn);

    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        // Pop
        curNode = travel.back().first;
        curDist = travel.back().second;
        travel.pop_back();

        // If the current node is the nearest neighbour
        if (curDist < nn.second)
        {
            nn.first = curNode;
            nn.second = curDist;
        }

        // Now push children in sorted order if potential NN among them
        unsigned num_children = unsigned(curNode->children.size());
        local_idx.resize(num_children);
        local_dists.resize(num_children);
        std::iota(local_idx.begin(), local_idx.end(), 0);
        for (unsigned i = 0; i < num_children; ++i)
            local_dists[i] = curNode->children[i]->dist(p);
        std::sort(std::begin(local_idx), std::end(local_idx), comp_x);

        const scalar best_dist_now = nn.second;
        for (const auto& child_idx : local_idx)
        {
            Node* child = curNode->children[child_idx];
            scalar dist_child = local_dists[child_idx];
            if (best_dist_now > dist_child - child->maxdistUB)
                travel.emplace_back(child, dist_child);
        }
    }
    return nn;
}


/****************************** k-Nearest Neighbours *************************************/

std::vector<std::pair<SGTree::Node*, scalar>> SGTree::kNearestNeighbours(const pointType &p, unsigned numNbrs) const
{
    // Do the worst initialization
    std::pair<SGTree::Node*, scalar> dummy(NULL, std::numeric_limits<scalar>::max());
    // List of k-nearest points till now
    std::vector<std::pair<SGTree::Node*, scalar>> nnList(numNbrs, dummy);

    // Iteration variables
    std::vector<std::pair<SGTree::Node*, scalar>> travel;
    SGTree::Node* curNode;
    scalar curDist;

    // Scratch memory
    std::vector<int> local_idx;
    std::vector<scalar> local_dists;
    auto comp_x = [&local_dists](int a, int b) { return local_dists[a] > local_dists[b]; };
    auto comp_pair = [](std::pair<SGTree::Node*, scalar> a, std::pair<SGTree::Node*, scalar> b) { return a.second < b.second; };

    // Initialize with root
    travel.emplace_back(root, root->dist(p));

    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        // Pop
        const auto current = travel.back();
        curNode = current.first;
        curDist = current.second;

        // TODO: efficient implementation ?
        // If the current node is eligible to get into the list
        if(curDist < nnList.back().second)
        {
            nnList.insert(
                std::upper_bound( nnList.begin(), nnList.end(), current, comp_pair ),
                current
            );
            nnList.pop_back();
        }
        travel.pop_back();

        // Now push children in sorted order if potential NN among them
        unsigned num_children = unsigned(curNode->children.size());
        local_idx.resize(num_children);
        local_dists.resize(num_children);
        std::iota(local_idx.begin(), local_idx.end(), 0);
        for (unsigned i = 0; i < num_children; ++i){
            local_dists[i] = curNode->children[i]->dist(p);
        }
        std::sort(local_idx.begin(), local_idx.end(), comp_x);

        const scalar best_dist_now = nnList.back().second;
        for (const auto& child_idx : local_idx)
        {
            Node* child = curNode->children[child_idx];
            scalar dist_child = local_dists[child_idx];
            if (best_dist_now > dist_child - child->maxdistUB)
                travel.emplace_back(child, dist_child);
        }
    }
    //std::cerr << "Done with one point" << std::endl;
    return nnList;
}


std::vector<std::pair<SGTree::Node*, scalar>> SGTree::kNearestNeighboursBeam(const pointType &p, unsigned numNbrs, unsigned beamSize) const
{
    // Do the worst initialization
    std::pair<SGTree::Node*, scalar> dummy(NULL, std::numeric_limits<scalar>::max());
    // List of k-nearest points till now
    std::vector<std::pair<SGTree::Node*, scalar>> nnList(numNbrs, dummy);

    // Iteration variables
    std::vector<std::pair<SGTree::Node*, scalar>> travel;

    SGTree::Node* curNode;
    scalar curDist;

    // Scratch memory
    std::vector<int> local_idx;
    std::vector<scalar> local_dists;
    auto comp_x = [&local_dists](int a, int b) { return local_dists[a] > local_dists[b]; };
    auto comp_pair = [](std::pair<SGTree::Node*, scalar> a, std::pair<SGTree::Node*, scalar> b) { return a.second < b.second; };

    // Initialize with root
    travel.emplace_back(root, root->dist(p));

    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        // Pop
        const auto current = travel.front();
        curNode = current.first;
        curDist = current.second;

        // TODO: efficient implementation ?
        // If the current node is eligible to get into the list
        if(curDist < nnList.back().second)
        {
            nnList.insert(
                std::upper_bound( nnList.begin(), nnList.end(), current, comp_pair ),
                current
            );
            nnList.pop_back();
        }
        travel.erase(travel.begin());

        // Now push children in sorted order if potential NN among them
        unsigned num_children = unsigned(curNode->children.size());
        local_idx.resize(num_children);
        local_dists.resize(num_children);
        std::iota(local_idx.begin(), local_idx.end(), 0);
        for (unsigned i = 0; i < num_children; ++i){
            local_dists[i] = curNode->children[i]->dist(p);
        }
        std::sort(local_idx.begin(), local_idx.end(), comp_x);

        const scalar best_dist_now = nnList.back().second;
        for (const auto& child_idx : local_idx)
        {
            Node* child = curNode->children[child_idx];
            scalar dist_child = local_dists[child_idx];
            if (best_dist_now > dist_child - child->maxdistUB) {
                // travel.emplace_back(child, dist_child);
                std::pair<SGTree::Node *, scalar>pair_to_add(child,dist_child);
                    travel.insert(
                        std::upper_bound( travel.begin(), travel.end(), pair_to_add, comp_pair),
                        pair_to_add
                    );
                    // std::cout << "[added to beam 2]  less than " << current->sepdist() << " beam size " << beam.size() << " checking kid " << current->children[child_idx]->UID << " dist " << dist_child << std::endl;

                    if (travel.size() > beamSize) {
                        travel.pop_back();
                    }

            }
        }
    }
    //std::cerr << "Done with one point" << std::endl;
    return nnList;
}

/****************************** Range Neighbours Search *************************************/

std::vector<std::pair<SGTree::Node*, scalar>> SGTree::rangeNeighbours(const pointType &p, scalar range) const
{
    // List of nearest neighbors in the range
    std::vector<std::pair<SGTree::Node*, scalar>> nnList;

    // Iteration variables
    std::vector<std::pair<SGTree::Node*, scalar>> travel;
    SGTree::Node* curNode;
    scalar curDist;

    // Scratch memory
    std::vector<int> local_idx;
    std::vector<scalar> local_dists;
    auto comp_x = [&local_dists](int a, int b) { return local_dists[a] > local_dists[b]; };

    // Initialize with root
    travel.emplace_back(root, root->dist(p));

    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        // Pop
        const auto current = travel.back();
        curNode = current.first;
        curDist = current.second;

        // If the current node is eligible to get into the list
        if (curDist < range)
        {
            nnList.push_back(current);
        }
        travel.pop_back();

        // Now push children in sorted order if potential NN among them
        unsigned num_children = unsigned(curNode->children.size());
        local_idx.resize(num_children);
        local_dists.resize(num_children);
        std::iota(local_idx.begin(), local_idx.end(), 0);
        for (unsigned i = 0; i < num_children; ++i)
            local_dists[i] = curNode->children[i]->dist(p);
        std::sort(local_idx.begin(), local_idx.end(), comp_x);

        for (const auto& child_idx : local_idx)
        {
            Node* child = curNode->children[child_idx];
            scalar dist_child = local_dists[child_idx];
            if (range > dist_child - child->maxdistUB)
                travel.emplace_back(child, dist_child);
        }
    }
    return nnList;
}


/****************************** Furthest Neighbour *************************************/

std::pair<SGTree::Node*, scalar> SGTree::FurthestNeighbour(const pointType &p) const
{
    std::pair<SGTree::Node*, scalar> fn(root, root->dist(p));
    std::vector<std::pair<SGTree::Node*, scalar>> travel;
    SGTree::Node* curNode;
    scalar curDist;

    // Scratch memory
    std::vector<int> local_idx;
    std::vector<scalar> local_dists;
    auto comp_x = [&local_dists](int a, int b) { return local_dists[a] < local_dists[b]; };

    // Initialize with root
    travel.push_back(fn);

    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        // Pop
        curNode = travel.back().first;
        curDist = travel.back().second;
        travel.pop_back();

        // If the current node is the nearest neighbour
        if (curDist > fn.second)
        {
            fn.first = curNode;
            fn.second = curDist;
        }

        // Now push children in sorted order if potential NN among them
        unsigned num_children = unsigned(curNode->children.size());
        local_idx.resize(num_children);
        local_dists.resize(num_children);
        std::iota(local_idx.begin(), local_idx.end(), 0);
        for (unsigned i = 0; i < num_children; ++i)
            local_dists[i] = curNode->children[i]->dist(p);
        std::sort(std::begin(local_idx), std::end(local_idx), comp_x);

        for (const auto& child_idx : local_idx)
        {
            Node* child = curNode->children[child_idx];
            scalar dist_child = local_dists[child_idx];
            if (fn.second < dist_child + child->maxdistUB)
                travel.emplace_back(child, dist_child);
        }
    }
    return fn;
}


/****************************** Serialization of Cover Trees *************************************/

// Pre-order traversal
char* SGTree::preorder_pack(char* buff, SGTree::Node* current) const
{
    // copy current node
    size_t shift = current->_p.rows() * sizeof(pointType::Scalar);
    char* start = (char*)current->_p.data();
    char* end = start + shift;
    std::copy(start, end, buff);
    buff += shift;

    shift = sizeof(int);
    start = (char*)&(current->level);
    end = start + shift;
    std::copy(start, end, buff);
    buff += shift;

    shift = sizeof(unsigned);
    start = (char*)&(current->ID);
    end = start + shift;
    std::copy(start, end, buff);
    buff += shift;

    shift = sizeof(unsigned);
    start = (char*)&(current->UID);
    end = start + shift;
    std::copy(start, end, buff);
    buff += shift;

    shift = sizeof(scalar);
    start = (char*)&(current->maxdistUB);
    end = start + shift;
    std::copy(start, end, buff);
    buff += shift;

#ifdef DEBUG
    std::cout << "Pre: " << current->ID << std::endl;
#endif

    // travrse children
    for (const auto& child : *current)
        buff = preorder_pack(buff, child);

    return buff;
}

// Post-order traversal
char* SGTree::postorder_pack(char* buff, SGTree::Node* current) const
{
    // travrse children
    for (const auto& child : *current)
        buff = postorder_pack(buff, child);

    // save current node ID
#ifdef DEBUG
    std::cout << "Post: " << current->ID << std::endl;
#endif
    size_t shift = sizeof(unsigned);
    char* start = (char*)&(current->ID);
    char* end = start + shift;
    std::copy(start, end, buff);
    buff += shift;

    return buff;
}

// reconstruct tree from Pre&Post traversals
void SGTree::PrePost(SGTree::Node*& current, char*& pre, char*& post)
{
    // The top element in preorder list PRE is the root of T
    current = new SGTree::Node();
    current->_p = pointType(D);
    for (unsigned i = 0; i < D; ++i)
    {
        current->_p[i] = *((pointType::Scalar *)pre);
        pre += sizeof(pointType::Scalar);
    }
    current->level = *((int *)pre);
    pre += sizeof(int);
    current->ID = *((unsigned *)pre);
    pre += sizeof(unsigned);
    current->UID = *((unsigned *)pre);
    pre += sizeof(unsigned);
    current->maxdistUB = *((scalar *)pre);
    pre += sizeof(scalar);

    // Construct subtrees until the root is found in the postorder list
    while (*((unsigned*)post) != current->ID)
    {
        SGTree::Node* temp = NULL;
        PrePost(temp, pre, post);
        current->children.push_back(temp);
    }

    //All subtrees of T are constructed
    post += sizeof(unsigned);       //Delete top element of POST
}

size_t SGTree::msg_size() const
{
    return 2 * sizeof(unsigned)
        + sizeof(pointType::Scalar)*D*N
        + sizeof(int)*N
        + sizeof(unsigned)*N*2
        + sizeof(scalar)*N
        + sizeof(unsigned)*N;
}

// Serialize to a buffer
char* SGTree::serialize() const
{
    //Covert following to char* buff with following order
    // N | D | (points, levels) | List
    char* buff = new char[msg_size()];

    char* pos = buff;

    // insert N
    unsigned shift = sizeof(unsigned);
    char* start = (char*)&(N);
    char* end = start + shift;
    std::copy(start, end, pos);
    pos += shift;

    // insert D
    shift = sizeof(unsigned);
    start = (char*)&(D);
    end = start + shift;
    std::copy(start, end, pos);
    pos += shift;

    // insert points and level
    pos = preorder_pack(pos, root);
    pos = postorder_pack(pos, root);

    //std::cout<<"Message size: " << msg_size() << ", " << pos - buff << std::endl;

    return buff;
}

// Deserialize from a buffer
void SGTree::deserialize(char* buff)
{
    /** Convert char* buff into following buff = N | D | (points, levels) | List **/
    //char* save = buff;

    // extract N and D
    N = *((unsigned *)buff);
    buff += sizeof(unsigned);
    D = *((unsigned *)buff);
    buff += sizeof(unsigned);

    //std::cout << "N: " << N << ", D: " << D << std::endl;

    // pointer to postorder list
    char* post = buff + sizeof(pointType::Scalar)*D*N
        + sizeof(int)*N + sizeof(unsigned)*N*2 + sizeof(scalar)*N;

    //reconstruction
    PrePost(root, buff, post);

    //delete[] save;
}

/****************************** Internal Constructors of Cover Trees *************************************/

//constructor: NULL tree
SGTree::SGTree(int truncate /*=-1*/ )
{
    root = NULL;
    min_scale = 1000;
    max_scale = 0;
    truncate_level = truncate;
    N = 0;
    D = 0;
}

//constructor: needs atleast 1 point to make a valid covertree
SGTree::SGTree(const pointType& p, int truncateArg /*=-1*/)
{
    min_scale = 1000;
    max_scale = 0;
    truncate_level = truncateArg;
    N = 1;
    D = unsigned(p.rows());

    root = new SGTree::Node;
    root->_p = p;
    root->ID = 0;
    root->UID = 0;
    root->level = 0;
    root->maxdistUB = 0;
}

//constructor: cover tree using points in the list between begin and end
SGTree::SGTree(const Eigen::Map<matrixType>& pMatrix, int truncateArg /*=-1*/, unsigned cores /*=true*/)
{
    size_t numPoints = pMatrix.cols();
    bool use_multi_core = cores > 1;
    this->cores = cores;

    //1. Compute the mean of entire data
    pointType mx = utils::ParallelAddMatrixNP(pMatrix).get_result()/(1.0*numPoints);

    //2. Compute distance of every point from the mean || Variance
    pointType dists = utils::ParallelDistanceComputeNP(pMatrix, mx).get_result();

    //3. argsort the distance to find approximate mediod
    std::vector<size_t> idx(numPoints);
    std::iota(std::begin(idx), std::end(idx), 0);
    auto comp_x = [&dists](size_t a, size_t b) { return dists[a] > dists[b]; };
    std::sort(std::begin(idx), std::end(idx), comp_x);
    std::cout<<"numPoints: " << numPoints << std::endl;
    std::cout<<"Max distance: " << dists[idx[0]] << std::endl;
    std::cout<<"Min distance: " << dists[idx[numPoints-1]] << std::endl;

    //4. Compute distance of every point from the mediod
    mx = pMatrix.col(idx[numPoints-1]);
    dists = utils::ParallelDistanceComputeNP(pMatrix, mx).get_result();
    scalar max_dist = dists.maxCoeff();

    int scale_val = int(std::ceil(std::log(max_dist)/std::log(base)));
    std::cout<<"Scale chosen: " << scale_val << std::endl;
    min_scale = scale_val; //-1000;
    max_scale = scale_val; //-1000;
    truncate_level = truncateArg;
    N = 1;
    D = unsigned(mx.rows());

    root = new SGTree::Node;
    root->_p = mx;
    root->level = scale_val; //-1000;
    root->maxdistUB = max_dist; // powdict[scale_val+1024];
    root->ID = 0;
    root->UID = idx[numPoints-1];

    std::cout << "(" << pMatrix.rows() << ", " << pMatrix.cols() << ")" << std::endl;
    if (use_multi_core)
    {
        if (50000 >= numPoints)
        {
            for (size_t i = 0; i < numPoints-1; ++i){
                // std::cout << "Insert i " << i << " idx[i] " << idx[i] << std::endl;
                utils::progressbar(i, numPoints);
                if(!insert(pMatrix.col(idx[i]), idx[i]))
                    std::cout << "Insert failed!!! " << idx[i] << std::endl;
            }
        }
        else
        {
            for (size_t i = 0; i < 50000; ++i){
                // std::cout << "Insert i " << i << " idx[i] " << idx[i] << std::endl;
                utils::progressbar(i, 50000);
                if(!insert(pMatrix.col(idx[i]), idx[i]))
                    std::cout << "Insert failed!!! " << idx[i] << std::endl;
            }
            utils::progressbar(50000, 50000);
            std::cerr<<std::endl;
            utils::parallel_for_progressbar(50000, numPoints-1, [&](size_t i)->void{
                // std::cout << "Insert i " << i << " idx[i] " << idx[i] << std::endl;
                if(!insert(pMatrix.col(idx[i]), idx[i]))
                    std::cout << "Insert failed!!! " << idx[i] << std::endl;
            }, cores);
        }
    }
    else
    {
        for (size_t i = 0; i < numPoints-1; ++i){
            utils::progressbar(i, numPoints);
            if(!insert(pMatrix.col(idx[i]), idx[i]))
                std::cout << "Insert failed!!! " << idx[i] <<  std::endl;
        }
    }
   // calc_maxdist();
   // print_stats();
}



//destructor: deallocating all memories by a post order traversal
SGTree::~SGTree()
{
    std::stack<SGTree::Node*> travel;

    if (root != NULL)
        travel.push(root);
    while (travel.size() > 0)
    {
        SGTree::Node* current = travel.top();
        travel.pop();

        for (const auto& child : *current)
        {
            if (child != NULL)
                travel.push(child);
        }

        delete current;
    }
}


/****************************** Public API for creation of Cover Trees *************************************/

//contructor: using matrix in col-major form!
SGTree* SGTree::from_matrix(const Eigen::Map<matrixType>& pMatrix, int truncate /*=-1*/, unsigned cores /*=true*/)
{
    std::cout << "SG Tree [v008] with base " << SGTree::base << std::endl;
    std::cout << "SG Tree with Number of Cores: " << cores << std::endl;
    SGTree* cTree = new SGTree(pMatrix, truncate, cores);
    return cTree;
}


/******************************************* Unit/Stat Testing ***************************************************/

bool SGTree::check_covering() const
{
    bool result = true;
    std::stack<SGTree::Node*> travel;
    SGTree::Node* curNode;

    // Initialize with root
    travel.push(root);

    // Pop, check and then push the children
    while (travel.size() > 0)
    {
        // Pop
        curNode = travel.top();
        travel.pop();

        // Check covering for the current -> children pair
        for (const auto& child : *curNode)
        {
            travel.push(child);
            if( curNode->dist(child) > curNode->covdist() )
                result = false;
            //std::cout << *curNode << " -> " << *child << " @ " << curNode->dist(child) << " | " << curNode->covdist() << std::endl;
        }
    }

    return result;
}


void SGTree::print_levels() const
{
    std::stack<SGTree::Node*> travel;
    std::map<int,unsigned> level_count;
    std::map<int,scalar> level_radius;
    std::map<int,scalar> level_max_radius;
    SGTree::Node* curNode;

    // Initialize with root
    travel.push(root);

    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        // Pop
        curNode = travel.top();
        travel.pop();

        // Count the level
        level_count[curNode->level]++;
        level_radius[curNode->level]+= curNode->maxdistUB;
        level_max_radius[curNode->level] = std::max(curNode->maxdistUB, level_max_radius[curNode->level]);

        // Now push the children
        for (const auto& child : *curNode)
            travel.push(child);
    }

    for(auto const& qc : level_count)
        std::cout << "Number of nodes at level "
                  << qc.first << " = " << qc.second
                  << " and avg radius " << level_radius[qc.first]/qc.second
                  << " and max radius " << level_max_radius[qc.first] << std::endl;
}


void SGTree::print_degrees() const
{
    std::stack<SGTree::Node*> travel;
    std::map<std::pair<int,int> ,unsigned> degree_count;
    SGTree::Node* curNode;

    // Initialize with root
    travel.push(root);

    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        // Pop
        curNode = travel.top();
        travel.pop();

        auto deg_level_pair = std::make_pair(curNode->level, curNode->children.size());

        // Count the level
        degree_count[deg_level_pair]++;


        // Now push the children
        for (const auto& child : *curNode)
            travel.push(child);
    }

    for(auto const& qc : degree_count)
        std::cout << "Number of nodes at level " << qc.first.first << " and degree "<< qc.first.second << " = " << qc.second << std::endl;
}


void SGTree::print_stats() const
{
    print_levels();
    print_degrees();
}


/******************************************* Ancilliary Functions ***************************************************/

// Getting the best K members
std::vector<unsigned> SGTree::getBestInitialPoints(unsigned numBest) const
{
    // Accumulated points
    std::vector<unsigned> allPoints;

    if (numBest > N)
      return allPoints;

    // Current level
    std::vector<SGTree::Node*> curLevel;
    curLevel.push_back(root);

    // Next level
    std::vector<SGTree::Node*> nextLevel;

    // Keep going down each level
    while (allPoints.size() < numBest)
    {
        // std::cout << "Size: " << curLevel.size() << std::endl;
        bool childLeft = false;
        for (const auto& member : curLevel)
        {
            allPoints.push_back(member->UID);
            if (member->children.size() > 0)
            {
                for (const auto& child : *member)
                {
                    childLeft = true;
                    nextLevel.push_back(child);
                }
            }
        }
        curLevel.swap(nextLevel);
        std::vector<Node*>().swap(nextLevel);
    }
    std::cout << "Found points: " << allPoints.size() << std::endl;

    return allPoints;
}

/******************************************* Pretty Print ***************************************************/

std::ostream& operator<<(std::ostream& os, const SGTree& ct)
{
    std::stack<SGTree::Node*> travel;
    SGTree::Node* curNode;

    // Initialize with root
    travel.push(ct.root);

    // Qualititively keep track of number of prints
    int numPrints = 0;
    // Pop, print and then push the children
    while (travel.size() > 0)
    {
        if (numPrints > 5000)
            std::cout << "Printing stopped prematurely, something wrong!";
        numPrints++;

        // Pop
        curNode = travel.top();
        travel.pop();

        // Print the current -> children pair
        for (const auto& child : *curNode)
            os << *curNode << " -> " << *child << std::endl;

        // Now push the children
        for (int i = int(curNode->children.size()) - 1; i >= 0; --i)
            travel.push(curNode->children[i]);
    }

    return os;
}
