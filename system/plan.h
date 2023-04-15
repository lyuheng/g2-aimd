#ifndef SYSTEM_PLAN_H
#define SYSTEM_PLAN_H

#include "common/meta.h"
#include "common/graph.h"

#include <queue>

class Plan
{
public:
    Plan() : hop_(0), root_degree_(0)
    {
    }

    int GetMinHopWithBFS(uintV root)
    {
        std::vector<bool> visited(vertex_count_, false);
        std::vector<int> level(vertex_count_, -1);
        int max_level_v = -1;
        int max_level_e = -1;

        // bfs from root vertex
        std::queue<uintV> queue;
        queue.push(root);
        visited[root] = true;
        level[root] = 0;
        while (!queue.empty())
        {
            uintV front = queue.front();
            queue.pop();
            max_level_v = level[front];
            for (size_t i = graph.GetRowPtrs()[front]; i < graph.GetRowPtrs()[front + 1]; ++i)
            {
                uintV j = graph.GetCols()[i];
                if (!visited[j])
                {
                    visited[j] = true;
                    level[j] = level[front] + 1;
                    queue.push(j);
                }
                else if (level[j] == level[front])
                {
                    max_level_e = level[j];
                }
            }
        }
        return max_level_v + (max_level_v == max_level_e ? 1 : 0);
    }

    void FindRoot()
    {
        vertex_count_ = graph.GetVertexCount();
        // only one here
        root_ = 0;
        hop_ = 10;
        for (uintV u = 0; u < vertex_count_; u++)
        {
            int tmp = GetMinHopWithBFS(u);
            if (tmp < hop_)
            {
                root_ = u;
                hop_ = tmp;
                root_degree_ = graph.GetRowPtrs()[u + 1] - graph.GetRowPtrs()[u];
            }
            else if (tmp == hop_ && root_degree_ < graph.GetRowPtrs()[u + 1] - graph.GetRowPtrs()[u])
            {
                root_ = u;
                hop_ = tmp;
                root_degree_ = graph.GetRowPtrs()[u + 1] - graph.GetRowPtrs()[u];
            }
        }
    }

    void GenerateSearchSequence()
    {
        // (hop, degree, id)
        std::vector<std::tuple<int, int, uintV>> weights(vertex_count_);
        std::vector<bool> visited(vertex_count_, false);
        visited[root_] = true;
        std::queue<uintV> queue;
        queue.push(root_);
        int hop = 0;
        weights[root_] = std::tuple<int, int, uintV>(hop, graph.GetRowPtrs()[root_ + 1] - graph.GetRowPtrs()[root_], root_);
        while (!queue.empty())
        {
            hop++;
            int size = queue.size();
            for (int i = 0; i < size; i++)
            {
                uintV front = queue.front();
                queue.pop();
                for (size_t i = graph.GetRowPtrs()[front]; i < graph.GetRowPtrs()[front + 1]; ++i)
                {
                    uintV j = graph.GetCols()[i];
                    if (!visited[j])
                    {
                        visited[j] = true;
                        queue.push(j);
                        weights[j] = std::tuple<int, int, uintV>(hop, graph.GetRowPtrs()[j + 1] - graph.GetRowPtrs()[j], j);
                    }
                }
            }
        }
        std::sort(weights.begin(), weights.end(), [](const auto& a, const auto& b) {
            if (std::get<0>(a) != std::get<0>(b))
                return std::get<0>(a) < std::get<0>(b);
            else if (std::get<1>(a) != std::get<1>(b))
                return std::get<1>(a) > std::get<1>(b);
            else if (std::get<2>(a) != std::get<2>(b))
                return std::get<2>(a) < std::get<2>(b);
            return false;
        });

        seq_.resize(vertex_count_);
        reverse_seq_.resize(vertex_count_);
        for (size_t i = 0; i < weights.size(); ++i)
        {
            uintV w = std::get<2>(weights[i]);
            seq_[i] = w;
            reverse_seq_[w] = i;
        }
    }

    void GenerateBackwardNeighbor()
    {
        std::vector<bool> visited(vertex_count_, false); 
    
        backNeighborCountHost = new uintV[vertex_count_];
        parentHost = new uintV[vertex_count_];
        std::fill(backNeighborCountHost, backNeighborCountHost + vertex_count_, 0);
        backNeighborsHost = new uintV[vertex_count_ * vertex_count_];

        visited[seq_[0]] = true;
        for (size_t i = 1; i < vertex_count_; ++i)
        {
            uintV vertex = seq_[i];
            for (uintV j = graph.GetRowPtrs()[vertex]; j < graph.GetRowPtrs()[vertex + 1]; ++j)
            {
                uintV nv = graph.GetCols()[j];
                if (visited[nv])
                {
                    backNeighborsHost[i * vertex_count_ + backNeighborCountHost[i]] = nv;
                    backNeighborCountHost[i]++;
                    parentHost[i] = nv;
                }
            }
            visited[vertex] = true;
        }

        share_intersection = new bool[vertex_count_];
        share_intersection[0] = false;
        

        for (size_t i = 1; i < vertex_count_; ++i)
        {
            if (backNeighborCountHost[i] > 2)
            {
                share_intersection[i] = true; // TODO: change this two part will be expanded only
                continue;
            }
            else if (backNeighborCountHost[i] == 2)
            {
                uintV last_vertex = seq_[i - 1];
                if (backNeighborsHost[i * vertex_count_] != last_vertex &&
                    backNeighborsHost[i * vertex_count_ + 1] != last_vertex)
                {
                    share_intersection[i] = true; // TODO: 
                    continue;
                }
            }
            share_intersection[i] = false;
        }

        // // if enabled: prefix-only 
        // share_intersection[1] = false;
        // for (size_t i = 2; i < vertex_count_; ++i)
        // {   
        //     share_intersection[i] = true;
        // }
    }

    void GeneratePreAfterBackwardNeighbor()
    {
        preBackNeighborCountHost = new uintV[vertex_count_];
        afterBackNeighborCountHost = new uintV[vertex_count_];
        std::fill(preBackNeighborCountHost, preBackNeighborCountHost + vertex_count_, 0);
        std::fill(afterBackNeighborCountHost, afterBackNeighborCountHost + vertex_count_, 0);

        preBackNeighborsHost = new uintV[vertex_count_ * vertex_count_];
        afterBackNeighborsHost = new uintV[vertex_count_ * vertex_count_];

        for (size_t i = 2; i < vertex_count_; ++i)
        {
            if (share_intersection[i])
            {
                uintV vertex = seq_[i];
                for (uintV j = graph.GetRowPtrs()[vertex]; j < graph.GetRowPtrs()[vertex + 1]; ++j) 
                {
                    uintV nv = graph.GetCols()[j];
                    if (reverse_seq_[nv] < reverse_seq_[vertex] - 1)
                    {
                        preBackNeighborsHost[i * vertex_count_ + preBackNeighborCountHost[i]] = nv;
                        preBackNeighborCountHost[i]++;
                    }
                    else if (reverse_seq_[nv] == reverse_seq_[vertex] - 1)
                    {
                        afterBackNeighborsHost[i * vertex_count_ + afterBackNeighborCountHost[i]] = nv;
                        afterBackNeighborCountHost[i]++;
                    }
                }
            }
        }
    }

    void GenerateUsefulOrder()
    {
        bool skip;
        condOrderHost = new uintV[vertex_count_ * vertex_count_ * 2];
        condNumHost = new uintV[vertex_count_];
        std::fill(condNumHost, condNumHost + vertex_count_, 0);

        preCondOrderHost = new uintV[vertex_count_ * vertex_count_ * 2];
        preCondNumHost = new uintV[vertex_count_];
        std::fill(preCondNumHost, preCondNumHost + vertex_count_, 0);

        afterCondOrderHost = new uintV[vertex_count_ * vertex_count_ * 2];
        afterCondNumHost = new uintV[vertex_count_];
        std::fill(afterCondNumHost, afterCondNumHost + vertex_count_, 0);

        for (size_t i = 0; i < vertex_count_; ++i)
        {
            size_t index = vertex_count_ * i * 2;
            for (size_t j = 0; j < vertex_count_; ++j)
            {
                if (reverse_seq_[i] > reverse_seq_[j])
                {
                    skip = false;
                    // check if there exists larger or less relationship
                    for (size_t k = 0; k < graph.order_[i].size(); ++k)
                    {
                        if (graph.order_[i][k].second == j)
                        {
                            condOrderHost[index] = graph.order_[i][k].first;
                            condOrderHost[index + 1] = j;
                            index += 2;
                            condNumHost[i] ++;
                            skip = true;
                            break;
                        }
                    }
                    if (!skip)
                    {
                        condOrderHost[index] = CondOperator::NON_EQUAL;
                        condOrderHost[index + 1] = j;
                        index += 2;
                        condNumHost[i] ++;
                    }
                }
            }
        }

        for (size_t i = 0; i < vertex_count_; ++i)
        {
            size_t index = vertex_count_ * i * 2;
            for (size_t j = 0; j < vertex_count_; ++j)
            {
                if (reverse_seq_[i] - 1 > reverse_seq_[j])
                {
                    skip = false;
                    // check if there exists larger or less relationship
                    for (size_t k = 0; k < graph.order_[i].size(); ++k)
                    {
                        if (graph.order_[i][k].second == j)
                        {
                            preCondOrderHost[index] = graph.order_[i][k].first;
                            preCondOrderHost[index + 1] = j;
                            index += 2;
                            preCondNumHost[i] ++;
                            skip = true;
                            break;
                        }
                    }
                    if (!skip)
                    {
                        preCondOrderHost[index] = CondOperator::NON_EQUAL;
                        preCondOrderHost[index + 1] = j;
                        index += 2;
                        preCondNumHost[i] ++;
                    }
                }
            }
        }
        for (size_t i = 0; i < vertex_count_; ++i)
        {
            size_t index = vertex_count_ * i * 2;
            for (size_t j = 0; j < vertex_count_; ++j)
            {
                if (reverse_seq_[i] - 1 == reverse_seq_[j])
                {
                    skip = false;
                    // check if there exists larger or less relationship
                    for (size_t k = 0; k < graph.order_[i].size(); ++k)
                    {
                        if (graph.order_[i][k].second == j)
                        {
                            afterCondOrderHost[index] = graph.order_[i][k].first;
                            afterCondOrderHost[index + 1] = j;
                            index += 2;
                            afterCondNumHost[i] ++;
                            skip = true;
                            break;
                        }
                    }
                    if (!skip)
                    {
                        afterCondOrderHost[index] = CondOperator::NON_EQUAL;
                        afterCondOrderHost[index + 1] = j;
                        index += 2;
                        afterCondNumHost[i] ++;
                    }
                }
            }
        }
    }

    void GenerateStoreStrategy()
    {
        strategy.resize(vertex_count_ + 1);

        // initial plan
        for (size_t i = 0; i < vertex_count_; ++i)
        {
            if (!share_intersection[i])
                strategy[i] = StoreStrategy::EXPAND;
            else
                strategy[i] = StoreStrategy::PREFIX;
        }
        strategy[vertex_count_] = StoreStrategy::COUNT;

        moving_lvl = new ui[vertex_count_];
        std::fill(moving_lvl, moving_lvl + vertex_count_, 1);

        // enumeration_cond = new uintV[vertex_count_ * vertex_count_ * 2];
        for (size_t i = 1; i < vertex_count_; ++i)
        {
            ui same_BN_cnt = 1; // including itself
            for (size_t j = i + 1; j < vertex_count_; ++j)
            {
                bool same_BN = true;
                // two consecutive vertices have the same backward neighbor
                if (backNeighborCountHost[i] == backNeighborCountHost[j])
                {
                    for (size_t k = 0; k < backNeighborCountHost[i]; ++k)
                    {
                        if (backNeighborsHost[i * vertex_count_ + k] != backNeighborsHost[j * vertex_count_ + k]) 
                        {
                            same_BN = false;
                            break;
                        }
                    }
                }
                else 
                    same_BN = false;
                if (same_BN)
                    same_BN_cnt += 1;
                else
                    break;
            }
            moving_lvl[i] = same_BN_cnt;
        }
    }

    int GetHop() { return hop_; }

    std::vector<uintV> GetSequence() { return seq_; }

    std::vector<uintV> GetReverseSequence() { return reverse_seq_; }


    Graph graph;
    size_t vertex_count_;
    size_t root_degree_;
    int hop_;
    uintV root_;
    std::vector<uintV> seq_;
    std::vector<ui> reverse_seq_;
    bool *share_intersection;

    uintV *backNeighborCountHost;
    uintV *backNeighborsHost;
    uintV *parentHost;
    uintV *condOrderHost;
    uintV *condNumHost;

    uintV *preBackNeighborCountHost;
    uintV *preBackNeighborsHost;
    uintV *preCondOrderHost;
    uintV *preCondNumHost;

    uintV *afterBackNeighborCountHost;
    uintV *afterBackNeighborsHost;
    uintV *afterCondOrderHost;
    uintV *afterCondNumHost;

    std::vector<StoreStrategy> strategy;

    ui *moving_lvl;

    uintV *enumeration_cond;
};



#endif