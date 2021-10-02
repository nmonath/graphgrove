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


#include "cover_tree.h"
#include <iostream>

scalar* CoverTree::compute_pow_table()
{
    scalar* powdict = new scalar[2048];
    for (int i = 0; i < 2048; ++i)
        powdict[i] = (scalar) pow(CoverTree::base, i - 1024);
    return powdict;
}

scalar* CoverTree::powdict = compute_pow_table();

#ifdef PRINTVER
std::map<int, std::atomic<unsigned>> CoverTree::Node::dist_count;
#endif

/******************************* Insert ***********************************************/
bool CoverTree::insert(CoverTree::Node* current, const pointType& p, unsigned UID, scalar dist_current)
{
    #ifdef DEBUG_INSERT
    std::cout << "insert(current=" << current->UID << ", uid=" << UID << ", dist_current=" << dist_current << ")" << std::endl << std::flush;
    std::cout << "root->level=" << current->level << std::endl << std::flush;
    #endif

    // global_mut.unlock_shared();
    // global_mut.lock();
    // we MUST be covered by current.
    assert(dist_current < current->covdist());

    // cover set Qi
    std::vector<std::pair<CoverTree::Node*, scalar>> Q_i(1, std::make_pair(current, dist_current));

    // Q = {Children(q) : q ∈ Qi}
    std::vector<std::pair<CoverTree::Node*, scalar>> Q;

    // who the parent of p will be.
    CoverTree::Node* parent = NULL;
    
    while(!Q_i.empty())
    {
        // Find a parent s.t. d(p, Q_i) ≤ 2^i
        // we take the closest one
        scalar dist_to_parent = std::numeric_limits<scalar>::max();
        #ifdef DEBUG_INSERT
        std::cout << "Q_i.size()=" << Q_i.size() << std::endl << std::flush;
        #endif
        for (const auto& q : Q_i)
        {
            #ifdef DEBUG_INSERT
            std::cout << "q.first=" << q.first->UID << " q.level=" << q.first->level << ", q.second=" << q.second << ", q.first->covdist()=" << q.first->covdist() << " dist_to_parent=" << dist_to_parent << " parent=" << parent << std::endl << std::flush;
            #endif
            if (q.second < dist_to_parent && q.second <= q.first->covdist()) 
            {
                parent = q.first;
                dist_to_parent = q.second;
            }
        }
        #ifdef DEBUG_INSERT
        std::cout << "dist_to_parent=" << dist_to_parent << ", parent=" << parent << std::endl << std::flush;
        #endif

        if (dist_to_parent == 0) {
            std::cout << "Duplicate entry!!!" << std::endl << std::flush;
            return false;
        }

        // Q = Qi−1 ={q∈Q: d(p,q) ≤ 2^i}
        Q.clear();

        // todo: parfor
        for (const auto& q : Q_i)
        {
            bool q_is_nested = false;
            #ifdef DEBUG_INSERT
            std::cout << "q=" << q.first->UID << " level=" << q.first->level << " sepdist" << q.first->sepdist() << " covdist=" << q.first->covdist() << std::endl << std::flush;
            #endif
            // todo use maxdisUB to skip kids??
            for(const auto& child : *(q.first))
            {
                // todo: check UIDs, if child.uid == q.uid
                // use q.second as dist_p_child
                scalar dist_p_child = child->dist(p);
                #ifdef DEBUG_INSERT
                std::cout << "child=" << child->UID << " child.level=" << child->level << " dist_p_child=" << dist_p_child << " sepdist=" << child->sepdist() << " covdist=" << child->covdist() << std::endl << std::flush;
                #endif
                if(dist_p_child <= q.first->covdist()/ (base-1)) {
                    Q.push_back(std::make_pair(child, dist_p_child));
                }
                q_is_nested = q_is_nested || child->UID == q.first->UID;
                #ifdef DEBUG_INSERT
                std::cout << "q_is_nested=" << child->UID << " dist_p_child=" << dist_p_child << " sepdist=" << child->sepdist() << " covdist=" << child->covdist() << std::endl << std::flush;
                #endif
            }
            #ifdef DEBUG_INSERT
            std::cout << "use_nesting " << use_nesting << " q_is_nested " << q_is_nested << " q.second " << q.second << " q.first->covdist())=" << q.first->covdist() << " q.first->sepdist()=" << q.first->sepdist() << std::endl << std::flush;
            std::cout << "Q.size() " << Q.size() << std::endl << std::flush;
            #endif

            if (use_nesting && !q_is_nested && q.second <= q.first->covdist()/ (base-1)) {
                #ifdef DEBUG_INSERT
                std::cout << "nesting!" << std::endl << std::flush;
                #endif
                // create a new child, a copy of the current node (with new id, but same UID)
                Node * new_child = q.first->setChild(q.first->_p, q.first->UID, N++);
                //  if (new_child->maxdistUB < q.second)
                //     new_child->maxdistUB = q.second;

                int local_min = min_scale.load();
                while( local_min > q.first->level - 1){
                    min_scale.compare_exchange_weak(local_min, q.first->level - 1, std::memory_order_relaxed, std::memory_order_relaxed);
                    local_min = min_scale.load();
                }
                Q.push_back(std::make_pair(new_child, q.second));
            }
            #ifdef DEBUG_INSERT
            std::cout << "Q.size() " << Q.size() << std::endl << std::flush;
            std::cout << "Q_i.size() " << Q_i.size() << std::endl << std::flush;
            #endif

        }
        std::swap(Q_i, Q);
        #ifdef DEBUG_INSERT
        std::cout << "Q.size() " << Q.size() << std::endl << std::flush;
        std::cout << "Q_i.size() " << Q_i.size() << std::endl << std::flush;
        #endif
    }
    #ifdef DEBUG_INSERT
    std::cout << "picked parent " << parent << std::endl << std::flush;
    std::cout << "parent level " << parent->level << std::endl << std::flush;
    #endif
    assert(parent != NULL);
    int new_id = N++;
    #ifdef DEBUG_INSERT
    std::cout << "setting child, parent->children.size()=" << parent->children.size() << std::endl << std::flush;
    #endif
    parent->setChild(p, UID, new_id);
    #ifdef DEBUG_INSERT
    std::cout << "set child! parent->children.size()=" << parent->children.size() << std::endl << std::flush;
    #endif
    int local_min = min_scale.load();
    while( local_min > parent->level - 1){
        min_scale.compare_exchange_weak(local_min, parent->level - 1, std::memory_order_relaxed, std::memory_order_relaxed);
        local_min = min_scale.load();
    }
    #ifdef DEBUG_INSERT
    std::cout << "local_min " << local_min << std::endl << std::flush;
    #endif 
    // global_mut.unlock();
    return true;
}


void CoverTree::calc_maxdist()
{
    std::vector<CoverTree::Node*> travel;
    std::vector<CoverTree::Node*> active;

    CoverTree::Node* current = root;

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

bool CoverTree::insert(const pointType& p, unsigned UID)
{
    global_mut.lock();
    bool result = false;
    id_valid = false;
    // global_mut.lock_shared();
    scalar root_dist_p = root->dist(p);
    if (root_dist_p > root->covdist())
    {
        // global_mut.unlock_shared();
        std::cout<<"Entered case 1: " << root->dist(p) << " " << root->covdist() << " " << root->level <<std::endl;
        std::cout<<"Requesting global lock!" <<std::endl;
        // global_mut.lock();
        
        while (root_dist_p > root->covdist())
        {
            CoverTree::Node* new_root = root->setParent(root->_p, root->UID, N++);
            std::cout<<"Entered case 1: " << root->dist(p) << " " << root->covdist() << " " << root->level << " new_root->level" << new_root->level <<std::endl;
            root = new_root;
            max_scale = root->level;
        }
        result = insert(root, p, UID, root->dist(p));
    } else {
        result = insert(root, p, UID, root->dist(p));
        // global_mut.unlock_shared();
    }
    global_mut.unlock();
    return result;
}

/******************************* Remove ***********************************************/

//TODO: Amortized implementation is needed


/****************************** Nearest Neighbour *************************************/

std::pair<CoverTree::Node*, scalar> CoverTree::NearestNeighbour(const pointType &p) const
{
    std::pair<CoverTree::Node*, scalar> nn(root, root->dist(p));
    std::vector<std::pair<CoverTree::Node*, scalar>> travel;
    CoverTree::Node* curNode;
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


void CoverTree::kNearestNeighbours(CoverTree::Node* current, scalar dist_current, const pointType& p, std::vector<std::pair<CoverTree::Node*, scalar>>& nnList) const
{
    // TODO: efficient implementation ?

    // If the current node is eligible to get into the list
    if(dist_current < nnList.back().second)
    {
        auto comp_x = [](std::pair<CoverTree::Node*, scalar> a, std::pair<CoverTree::Node*, scalar> b) { return a.second < b.second; };
        std::pair<CoverTree::Node*, scalar> temp(current, dist_current);
        nnList.insert(
            std::upper_bound( nnList.begin(), nnList.end(), temp, comp_x ),
            temp
        );
        nnList.pop_back();
    }

    // Sort the children
    unsigned num_children = current->children.size();
    std::vector<int> idx(num_children);
    std::iota(std::begin(idx), std::end(idx), 0);
    std::vector<scalar> dists(num_children);
    for (unsigned i = 0; i < num_children; ++i)
        dists[i] = current->children[i]->dist(p);
    auto comp_x = [&dists](int a, int b) { return dists[a] < dists[b]; };
    std::sort(std::begin(idx), std::end(idx), comp_x);

    for (const auto& child_idx : idx)
    {
        Node* child = current->children[child_idx];
        scalar dist_child = dists[child_idx];
        if ( nnList.back().second > dist_child - child->maxdistUB)
            kNearestNeighbours(child, dist_child, p, nnList);
    }
}

std::vector<std::pair<CoverTree::Node*, scalar>> CoverTree::kNearestNeighbours(const pointType &queryPt, unsigned numNbrs) const
{
    // Do the worst initialization
    std::pair<CoverTree::Node*, scalar> dummy(new CoverTree::Node(), std::numeric_limits<scalar>::max());
    // List of k-nearest points till now
    std::vector<std::pair<CoverTree::Node*, scalar>> nnList(numNbrs, dummy);

    // Call with root
    scalar dist_root = root->dist(queryPt);
    kNearestNeighbours(root, dist_root, queryPt, nnList);

    return nnList;
}


/****************************** Range Neighbours Search *************************************/

std::vector<std::pair<CoverTree::Node*, scalar>> CoverTree::rangeNeighbours(const pointType &p, scalar range) const
{
    // List of nearest neighbors in the range
    std::vector<std::pair<CoverTree::Node*, scalar>> nnList;

    // Iteration variables
    std::vector<std::pair<CoverTree::Node*, scalar>> travel;
    CoverTree::Node* curNode;
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

std::pair<CoverTree::Node*, scalar> CoverTree::FurthestNeighbour(const pointType &p) const
{
    std::pair<CoverTree::Node*, scalar> fn(root, root->dist(p));
    std::vector<std::pair<CoverTree::Node*, scalar>> travel;
    CoverTree::Node* curNode;
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
char* CoverTree::preorder_pack(char* buff, CoverTree::Node* current) const
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
char* CoverTree::postorder_pack(char* buff, CoverTree::Node* current) const
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
void CoverTree::PrePost(CoverTree::Node*& current, char*& pre, char*& post)
{
    // The top element in preorder list PRE is the root of T
    current = new CoverTree::Node();
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
        CoverTree::Node* temp = NULL;
        PrePost(temp, pre, post);
        current->children.push_back(temp);
    }

    //All subtrees of T are constructed
    post += sizeof(unsigned);       //Delete top element of POST
}

size_t CoverTree::msg_size() const
{
    return 2 * sizeof(unsigned)
        + sizeof(pointType::Scalar)*D*N
        + sizeof(int)*N
        + sizeof(unsigned)*N*2
        + sizeof(scalar)*N
        + sizeof(unsigned)*N;
}

// Serialize to a buffer
char* CoverTree::serialize() const
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
void CoverTree::deserialize(char* buff)
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
CoverTree::CoverTree(int truncate /*=-1*/ )
{
    root = NULL;
    min_scale = 1000;
    max_scale = 0;
    truncate_level = truncate;
    N = 0;
    D = 0;
}

//constructor: needs atleast 1 point to make a valid covertree
CoverTree::CoverTree(const pointType& p, int truncateArg /*=-1*/)
{
    min_scale = 1000;
    max_scale = 0;
    truncate_level = truncateArg;
    N = 1;
    D = unsigned(p.rows());

    root = new CoverTree::Node;
    root->_p = p;
    root->ID = 0;
    root->UID = 0;
    root->level = 0;
    root->maxdistUB = 0;
}

//constructor: cover tree using points in the list between begin and end
CoverTree::CoverTree(const Eigen::Map<matrixType>& pMatrix, int truncateArg /*=-1*/, unsigned cores /*=true*/)
{
    size_t numPoints = pMatrix.cols();
    bool use_multi_core = cores != 0;
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

    root = new CoverTree::Node;
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
                utils::progressbar(i, numPoints);
                if(!insert(pMatrix.col(idx[i]), idx[i]))
                    std::cout << "Insert failed!!! " << idx[i] << std::endl;
            }
        }
        else
        {
            for (size_t i = 0; i < 50000; ++i){
                utils::progressbar(i, 50000);
                if(!insert(pMatrix.col(idx[i]), idx[i]))
                    std::cout << "Insert failed!!! " << idx[i] << std::endl;
            }
            utils::progressbar(50000, 50000);
            std::cerr<<std::endl;
            utils::parallel_for_progressbar(50000, numPoints-1, [&](size_t i)->void{
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
CoverTree::~CoverTree()
{
    std::stack<CoverTree::Node*> travel;

    if (root != NULL)
        travel.push(root);
    while (travel.size() > 0)
    {
        CoverTree::Node* current = travel.top();
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
CoverTree* CoverTree::from_matrix(const Eigen::Map<matrixType>& pMatrix, int truncate /*=-1*/, unsigned cores /*=true*/)
{
    std::cout << "Cover Tree with base " << CoverTree::base << std::endl;
    std::cout << "Cover Tree with Number of Cores: " << cores << std::endl;
    CoverTree* cTree = new CoverTree(pMatrix, truncate, cores);
    return cTree;
}


/******************************************* Unit/Stat Testing ***************************************************/

bool CoverTree::check_covering() const
{
    bool result = true;
    std::stack<CoverTree::Node*> travel;
    CoverTree::Node* curNode;

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
            if( curNode->dist(child) > curNode->covdist() ) {
                result = false;
                std::cout << curNode->UID << " -> " << child->UID << " @ " << curNode->dist(child) << " | " << curNode->covdist() << std::endl;
            }
            
        }
    }

    return result;
}


bool CoverTree::check_nesting() const
{
	bool result = true;
	std::stack<CoverTree::Node*> travel;
	CoverTree::Node* curNode;
	std::set<int> failed;
	std::set<int> levels;

	// Initialize with root
	travel.push(root);

	// Pop, check and then push the children
	while (!travel.empty())
	{
		// Pop
		curNode = travel.top();
		travel.pop();
		bool found = true;

        // nesting is implicit
        // nesting fails when we have a node
        // with a child that violates separation 

		// Check covering for the current -> children pair
		for (const auto& child : *curNode)
		{
			travel.push(child);
			if( child->UID != curNode->UID && curNode->dist(child) < child->covdist() ) {
				found = false;
				break;
                std::cout << "Nesting check " << curNode->UID << " -> " << child->UID << " @ " << curNode->dist(child) << " | " << curNode->covdist() << " | " << curNode->sepdist() << std::endl;
			}
		}
		if (!found) {
			result = false;
			failed.insert(curNode->level);
			// std::cout << "Nesting check failed at level " << curNode->level << std::endl;
		}
		levels.insert(curNode->level);
	}
	std::cout << "Nesting check " << result << ". Failed at " << failed.size() << std::endl;
	for (const int level : failed) {
		std::cout << "Nesting failed at level " << level << std::endl;
	}

	return result;
}

void CoverTree::print_levels() const
{
    std::stack<CoverTree::Node*> travel;
    std::map<int,unsigned> level_count;
    std::map<int,scalar> level_radius;
    std::map<int,scalar> level_max_radius;
    CoverTree::Node* curNode;

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


void CoverTree::print_degrees() const
{
    std::stack<CoverTree::Node*> travel;
    std::map<std::pair<int,int> ,unsigned> degree_count;
    CoverTree::Node* curNode;

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


void CoverTree::print_stats() const
{
    print_levels();
    print_degrees();
}


/******************************************* Ancilliary Functions ***************************************************/

// Getting the best K members
std::vector<unsigned> CoverTree::getBestInitialPoints(unsigned numBest) const
{
    // Accumulated points
    std::vector<unsigned> allPoints;

    if (numBest > N)
      return allPoints;

    // Current level
    std::vector<CoverTree::Node*> curLevel;
    curLevel.push_back(root);

    // Next level
    std::vector<CoverTree::Node*> nextLevel;

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

std::ostream& operator<<(std::ostream& os, const CoverTree& ct)
{
    std::stack<CoverTree::Node*> travel;
    CoverTree::Node* curNode;

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