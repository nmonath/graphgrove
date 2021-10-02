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

# ifndef _COVER_TREE_H
# define _COVER_TREE_H

#include <Eigen/Core>
#include <atomic>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <shared_mutex>
#include <stack>
#include <string>
#include <vector>
#include <set>

#include "utils.h"

class CoverTree
{
/************************* Internal Functions ***********************************************/
protected:
    /*** Base to use for the calculations ***/
    static constexpr scalar base = 1.3;
    static scalar* compute_pow_table();
    static scalar* powdict;
    unsigned cores = -1;
    bool use_nesting = true;

public:
    /*** structure for each node ***/
    struct Node
    {
        pointType _p;                       // point associated with the node
        std::vector<Node*> children;        // list of children
        int level;                          // current level of the node
        scalar maxdistUB;                   // upper bound of distance to any of descendants
        unsigned ID;                        // mutable ID of current node
        unsigned UID;                       // external unique ID for current node
        std::string ext_prop;               // external encoded propertoes of current node

        mutable std::shared_timed_mutex mut;// lock for current node

        #ifdef PRINTVER
        static std::map<int,std::atomic<unsigned>> dist_count;
        #endif

        /*** Node modifiers ***/
        scalar covdist() const                   // covering distance of subtree at current node
        {
            return powdict[level + 1024];
        }
        scalar sepdist() const                   // separating distance between nodes at current level
        {
            return powdict[level + 1023];
        }
        scalar dist(const pointType& pp) const   // L2 distance between current node and point pp
        {
            #ifdef PRINTVER
            dist_count[level].fetch_add(1, std::memory_order_relaxed);
            #endif
            return (_p - pp).norm();
        }


        scalar dist(const Node* n) const         // L2 distance between current node and node n
        {
            #ifdef PRINTVER
            dist_count[level].fetch_add(1, std::memory_order_relaxed);
            #endif
            return (_p - n->_p).norm();
        }
        Node* setChild(const pointType& pIns,    // insert a new child of current node with point pIns
                       unsigned UID = 0,
                       int new_id=-1)
        {
            Node* temp = new Node;
            temp->_p = pIns;
            temp->level = level - 1;
            temp->maxdistUB = 0; // powdict[level + 1024];
            temp->ID = new_id;
            temp->UID = UID;
            children.push_back(temp);
            return temp;
        }

        Node* setParent(const pointType& pIns,    // insert a new child of current node with point pIns
                       unsigned UID = 0,
                       int new_id=-1)
        {
            Node* temp = new Node;
            temp->_p = pIns;
            temp->level = level + 1;
            temp->maxdistUB = 0; // powdict[level + 1024];
            temp->ID = new_id;
            temp->UID = UID;
            temp->children.push_back(this);
            return temp;
        }

        /*** erase child ***/
        void erase(size_t pos)
        {
            children[pos] = children.back();
            children.pop_back();
        }

        void erase(std::vector<Node*>::iterator pos)
        {
            *pos = children.back();
            children.pop_back();
        }

        /*** Iterator access ***/
        inline std::vector<Node*>::iterator begin()
        {
            return children.begin();
        }
        inline std::vector<Node*>::iterator end()
        {
            return children.end();
        }
        inline std::vector<Node*>::const_iterator begin() const
        {
            return children.begin();
        }
        inline std::vector<Node*>::const_iterator end() const
        {
            return children.end();
        }

        /*** Pretty print ***/
        friend std::ostream& operator<<(std::ostream& os, const Node& ct)
        {
            if (ct._p.rows() < 6)
            {
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
                os << "(" << ct._p.format(CommaInitFmt) << ":" << ct.level << ":" << ct.maxdistUB <<  ":" << ct.ID << ")";
            }
            else
            {
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
                os << "([" << ct._p.head<3>().format(CommaInitFmt) << ", ..., " << ct._p.tail<3>().format(CommaInitFmt) << "]:" << ct.level << ":" << ct.maxdistUB <<  ":" << ct.ID << ")";
            }
            return os;
        }
    };

protected:
    Node* root;                         // Root of the tree
    std::atomic<int> min_scale;         // Minimum scale
    std::atomic<int> max_scale;         // Minimum scale
    int truncate_level;                 // Relative level below which the tree is truncated
    bool id_valid;

    std::atomic<unsigned> N;            // Number of points in the cover tree
    unsigned D;                         // Dimension of the points

    std::shared_timed_mutex global_mut;	// lock for changing the root

    /*** Insertion helper function ***/
    bool insert(Node* current, const pointType& p, unsigned UID, scalar dist_current);

    /*** kNN helper function ***/
    void kNearestNeighbours(Node* current, scalar dist_current, const pointType& p, std::vector<std::pair<Node*, scalar>>& nnList) const;

    /*** Serialize/Desrialize helper function ***/
    char* preorder_pack(char* buff, Node* current) const;       // Pre-order traversal
    char* postorder_pack(char* buff, Node* current) const;      // Post-order traversal
    void PrePost(Node*& current, char*& pre, char*& post);

public:
    /*** Internal Contructors ***/
    /*** Constructor: needs atleast 1 point to make a valid covertree ***/
    // NULL tree
    CoverTree(int truncate = -1);
    // cover tree with one point as root
    CoverTree(const pointType& p, int truncate = -1);
    // cover tree using points in the list between begin and end
    CoverTree(const Eigen::Map<matrixType>& pMatrix, int truncate = -1, unsigned cores = -1);

    /*** Destructor ***/
    /*** Destructor: deallocating all memories by a post order traversal ***/
    ~CoverTree();

/************************* Public API ***********************************************/
    /*** Construct cover tree using all points in the matrix in row-major form ***/
    static CoverTree* from_matrix(const Eigen::Map<matrixType>& pMatrix, int truncate = -1, unsigned cores = -1);

    /*** Get root ***/
    Node* get_root() {return root;}

    /*** Insert point p into the cover tree ***/
    bool insert(const pointType& p, unsigned UID);

    /*** Remove point p into the cover tree ***/
    bool remove(const pointType& p) {return false;}

    /*** Nearest Neighbour search ***/
    std::pair<CoverTree::Node*, scalar> NearestNeighbour(const pointType &p) const;

    /*** k-Nearest Neighbour search ***/
    std::vector<std::pair<CoverTree::Node*, scalar>> kNearestNeighbours(const pointType &p, unsigned k = 10) const;

    /*** Range search ***/
    std::vector<std::pair<CoverTree::Node*, scalar>> rangeNeighbours(const pointType &queryPt, scalar range = 1.0) const;

    /*** Furthest Neighbour search ***/
    std::pair<CoverTree::Node*, scalar> FurthestNeighbour(const pointType &p) const;

    /*** Serialize/Desrialize: useful for Pickling ***/
    char* serialize() const;                                    // Serialize to a buffer
    size_t msg_size() const;
    void deserialize(char* buff);                               // Deserialize from a buffer

    /*** Unit Tests ***/
    bool check_covering() const;
    bool check_nesting() const;
    void print_stats() const;
    void print_levels() const;
    void print_degrees() const;

    /*** Some spread out points in the space ***/
    std::vector<unsigned> getBestInitialPoints(unsigned numBest) const;

    /*** Pretty print ***/
    friend std::ostream& operator<<(std::ostream& os, const CoverTree& ct);

    void calc_maxdist();
    int get_tree_size() {return N.load();}
};

#endif  // _COVER_TREE_H
