#ifndef APP_GMATCH_GMATCH_H
#define APP_GMATCH_GMATCH_H

#include "system/appbase.h"
#include "system/buffer.h"
#include "common/meta.h"
#include "system/util.h"

#define SHM_CAP 350
#define BATCH_SIZE 100000000


// 2634376 * 4 = 10537504 2634376

class GMatchApp : public AppBase<BufferBase>
{
public:
    // temporary array to store local candidate ?????
    ui *tempv;
    // bool *templ;

    ui *pre_intersection;

    void allocateMemory()
    {
        // AppBase<BufferBase>::allocateMemory();

        allocateGMMemory();
        chkerr(cudaMalloc((void **)&tempv, TEMPSIZE * N_WARPS * sizeof(ui)));
        // chkerr(cudaMalloc((void **)&templ, TEMPSIZE * N_WARPS * sizeof(bool)));
        chkerr(cudaMalloc((void **)&pre_intersection, TEMPSIZE * N_WARPS * sizeof(ui)));
    }

    __device__ ui writeToTemp(ui v, ui l, bool pred, unsigned int sglen)
    {
        unsigned int loc = scanIndex(pred) + sglen;
        // popc gives inclusive sum scan, subtract pred to make it exclusive
        // add sglen to find exact location in the temp
        assert(loc < TEMPSIZE);
        if (pred)
        {
            this->tempv[loc + GLWARPID * TEMPSIZE] = v;
            // this->templ[loc + GLWARPID * TEMPSIZE] = l;
        }
        if (LANEID == 31) // last lane's loc+pred is number of items found in this scan
            sglen = loc + pred;
        sglen = __shfl_sync(FULL, sglen, 31);
        return sglen;
    }

    __device__ ui writeToPreIntersection(ui v, bool pred, unsigned int sglen)
    {
        unsigned int loc = scanIndex(pred) + sglen;
        // popc gives inclusive sum scan, subtract pred to make it exclusive
        // add sglen to find exact location in the temp
        assert(loc < TEMPSIZE);
        if (pred)
        {
            this->pre_intersection[loc + GLWARPID * TEMPSIZE] = v;
        }
        if (LANEID == 31) // last lane's loc+pred is number of items found in this scan
            sglen = loc + pred;
        sglen = __shfl_sync(FULL, sglen, 31);
        return sglen;
    }

    __device__ void generateSubgraphs(ui base)
    {   
        for (ui i = GLWARPID; i < sg->chunk[0]; i += N_WARPS)
        {
            ui v = base + i;
        
            if (v >= ctx->sources_num[0])
                break;
            unsigned int vt = sg->append(1); // allocates a subgraph by atomic operations, and puts v as well
            if (LANEID == 0)
            {
                sg->wrBuff.vertices[vt] = ctx->sources[v];
            }
        }
    }

    __device__ void processSubgraphs()
    {
        StoreStrategy CUR_MODE, NEXT_MODE;

        __shared__ ui partial_subgraphs[WARPS_EACH_BLK][8];

        size_t thread_count = 0;

        while (true)
        {
            SubgraphOffsets so = sg->next();

            if (sg->isEnd(so.st))
                break;

            if (so.md == 0)
                CUR_MODE = StoreStrategy::EXPAND;
            else
                CUR_MODE = StoreStrategy::PREFIX;
            
            if (sg->isOverflow())
                break;

            if (sg->isOverflowToHost())
            {
                dumpToHost(&so);
                continue;
            }

            ui id, sglen;
            if (CUR_MODE == StoreStrategy::EXPAND)
            {
                id = so.en - so.st;
                sglen = so.en - so.st;
            }
            else if (CUR_MODE == StoreStrategy::PREFIX)
            {
                id = so.md - so.st + 1;
                sglen = so.md - so.st + 1;
            }

            NEXT_MODE = strategy[id + 1];
            ui u = matchOrder[id];

            
            /**
            if (CUR_MODE == StoreStrategy::EXPAND && NEXT_MODE != StoreStrategy::PREFIX)
            {
                if (so.st + LANEID < so.en)
                    partial_subgraphs[WARPID][LANEID] = sg->rdBuff.vertices[so.st + LANEID];
                __syncwarp();
                
                // select the pivot with least # of candidates
                ui bnCount = backNeighborCount[id];
                ui u_prime = backNeighbors[querySize[0] * id];
                ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                uintE u_prime_M_st = ctx->d_row_ptrs[u_prime_M];
                uintE u_prime_M_en = ctx->d_row_ptrs[u_prime_M + 1];
                uintE min = u_prime_M_en - u_prime_M_st;
                ui parent_u = u_prime;
                for (ui i = 1; i < bnCount; ++i)
                {
                    ui u_prime = backNeighbors[querySize[0] * id + i];
                    ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                    uintE u_prime_M_st = ctx->d_row_ptrs[u_prime_M];
                    uintE u_prime_M_en = ctx->d_row_ptrs[u_prime_M + 1];
                    uintE neigh_len = u_prime_M_en - u_prime_M_st;
                    if (neigh_len < min)
                    {
                        min = neigh_len;
                        parent_u = u_prime;
                    }
                }
                ui parent_u_M = partial_subgraphs[WARPID][ID2order[parent_u]];

                uintE pu_st = ctx->d_row_ptrs[parent_u_M];
                uintE pu_en = ctx->d_row_ptrs[parent_u_M + 1];

                bool pred;
                ui condCount = condNum[u];
                ui vertex;
                uintE base_i = pu_st;

                do {
                    ui len = 0;
                    for ( ; base_i < pu_en; base_i += 32)
                    {
                        uintE il = base_i + LANEID;
                        pred = il < pu_en;

                        if (pred)
                        {
                            vertex = ctx->d_cols[il];
                            for (ui j = 0; j < bnCount; ++j)
                            {
                                ui u_prime = backNeighbors[querySize[0] * id + j];
                                if (u_prime == parent_u)
                                    continue;
                                ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                                uintE u_prime_M_st = ctx->d_row_ptrs[u_prime_M];
                                uintE u_prime_M_en = ctx->d_row_ptrs[u_prime_M + 1];

                                pred = binarySearch(ctx->d_cols, u_prime_M_st, u_prime_M_en, vertex);
                                if (!pred) break;
                            }
            
                            if (pred)
                            {
                                for (ui k = 0; k < condCount; ++k)
                                {
                                    ui cond = condOrder[u * querySize[0] * 2 + 2 * k];
                                    ui cond_vertex = condOrder[u * querySize[0] * 2 + 2 * k + 1];
                                    ui cond_vertex_M = partial_subgraphs[WARPID][ID2order[cond_vertex]];
                                    if (cond == CondOperator::LESS_THAN)
                                    {
                                        if (cond_vertex_M <= vertex)
                                        {
                                            pred = false; 
                                            break;
                                        }
                                    }
                                    else if (cond == CondOperator::LARGER_THAN)
                                    {
                                        if (cond_vertex_M >= vertex)
                                        {
                                            pred = false; 
                                            break;
                                        }
                                    }
                                    else if (cond == CondOperator::NON_EQUAL)
                                    {
                                        if (cond_vertex_M == vertex)
                                        {
                                            pred = false; 
                                            break;
                                        }
                                    }
                                }
                                if (pred)
                                {
                                    if (sglen + 1 == querySize[0])
                                    {
                                        thread_count += 1;
                                    }
                                }
                            }
                        }
                        if (sglen + 1 != querySize[0])
                        {
                            // ui val = pred ? vertex : 0;
                            unsigned int loc = scanIndex(pred) + len;
                            assert(loc < SHM_CAP);
                            if (pred)
                            {
                                shm_temp[WARPID][loc] = vertex;
                            }
                            if (LANEID == 31) // last lane's loc+pred is number of items found in this scan
                                len = loc + pred;
                            len = __shfl_sync(FULL, len, 31);
                            
                            if (len >= SHM_CAP - 32)
                            { 
                                // if (GTHID == 0) printf("* "); 
                                base_i += 32;
                                break;
                            }
                        }
                    }

                    auto vt = sg->append_batch(sglen + 1, len, StoreStrategy::EXPAND);
                    if (sg->isOverflow())
                        return;
                    for (ui i = LANEID; i < len; i += 32)
                    {
                        for (ui j = 0; j < sglen; ++j)
                        {
                            auto k = vt + i * (sglen + 1) + j;
                            sg->wrBuff.vertices[k] = partial_subgraphs[WARPID][j];
                        }
                        sg->wrBuff.vertices[vt + i * (sglen + 1) + sglen] = shm_temp[WARPID][i]; // add q on the back
                    }
                } while (base_i < pu_en);
                continue;
            }
            **/
            
        
 
            // do pre-intersection here

            if (shareIntersection[id])
            {
                if (so.st + LANEID < so.md)
                    partial_subgraphs[WARPID][LANEID] = sg->rdBuff.vertices[so.st + LANEID];
                __syncwarp();

                ui bnCount = preBackNeighborCount[id];
                ui u_prime = preBackNeighbors[querySize[0] * id];
                ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                uintE u_prime_M_st = ctx->d_row_ptrs[u_prime_M];
                uintE u_prime_M_en = ctx->d_row_ptrs[u_prime_M + 1];
                uintE min = u_prime_M_en - u_prime_M_st;
                ui parent_u = u_prime;
                for (ui i = 1; i < bnCount; ++i)
                {
                    ui u_prime = preBackNeighbors[querySize[0] * id + i];
                    ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                    uintE u_prime_M_st = ctx->d_row_ptrs[u_prime_M];
                    uintE u_prime_M_en = ctx->d_row_ptrs[u_prime_M + 1];
                    uintE neigh_len = u_prime_M_en - u_prime_M_st;
                    if (neigh_len < min)
                    {
                        min = neigh_len;
                        parent_u = u_prime;
                    }
                }
            

                ui parent_u_M = partial_subgraphs[WARPID][ID2order[parent_u]];
                uintE pu_st = ctx->d_row_ptrs[parent_u_M];
                uintE pu_en = ctx->d_row_ptrs[parent_u_M + 1];
                ui len = 0;
                bool pred;
                ui condCount = preCondNum[u];
                ui vertex;
                uintE base_i = pu_st;

                do {
                    len = 0;

                    for ( ; base_i < pu_en; base_i += 32)
                    {
                        uintE il = base_i + LANEID;
                        pred = il < pu_en;

                        if (pred)
                        {
                            vertex = ctx->d_cols[il];

                            for (ui k = 0; k < condCount; ++k)
                            {
                                ui cond = preCondOrder[u * querySize[0] * 2 + 2 * k];
                                ui cond_vertex = preCondOrder[u * querySize[0] * 2 + 2 * k + 1];
                                ui cond_vertex_M = partial_subgraphs[WARPID][ID2order[cond_vertex]];
                                if (cond == CondOperator::LESS_THAN)
                                {
                                    if (cond_vertex_M <= vertex)
                                    {
                                        pred = false;
                                        break;
                                    }
                                }
                                else if (cond == CondOperator::LARGER_THAN)
                                {
                                    if (cond_vertex_M >= vertex)
                                    {
                                        pred = false;
                                        break;
                                    }
                                }
                                else if (cond == CondOperator::NON_EQUAL)
                                {
                                    if (cond_vertex_M == vertex)
                                    {
                                        pred = false;
                                        break;
                                    }
                                }
                            }

                            if (pred)
                            {
                                for (ui j = 0; j < bnCount; ++j)
                                {
                                    ui u_prime = preBackNeighbors[querySize[0] * id + j];
                                    if (u_prime == parent_u)
                                        continue;
                                    ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                                    uintE u_prime_M_st = ctx->d_row_ptrs[u_prime_M];
                                    uintE u_prime_M_en = ctx->d_row_ptrs[u_prime_M + 1];
                                    pred = binarySearch(ctx->d_cols, u_prime_M_st, u_prime_M_en, vertex);
                                    if (!pred) break;
                                }
                            }
                        }
                        ui val = pred ? vertex : 0;
                        len = writeToPreIntersection(val, pred, len);

                        if (len >= TEMPSIZE - 32)
                        { 
                            if (GTHID == 0) printf("* ");
                            base_i += 32;
                            break;
                        }
                    }
                    ui pre_len = len;

                    for (ui subgraph_id = so.md; subgraph_id < so.en; ++subgraph_id)
                    {

                        if (LANEID == 0)
                        {
                            partial_subgraphs[WARPID][so.md - so.st] = sg->rdBuff.vertices[subgraph_id];
                        }
                        __syncwarp();

                        ui len = 0;
                        bool pred;
                        ui condCount = afterCondNum[u];
                        ui bnCount = afterBackNeighborCount[id];
                        ui vertex;

        
                        for (uintE i = 0; i < pre_len; i += 32)
                        {
                            uintE il = i + LANEID;
                            pred = il < pre_len;

                            if (pred)
                            {
                                vertex = pre_intersection[GLWARPID * TEMPSIZE + il];
                                for (ui k = 0; k < condCount; ++k)
                                {
                                    ui cond = afterCondOrder[u * querySize[0] * 2 + 2 * k];
                                    ui cond_vertex = afterCondOrder[u * querySize[0] * 2 + 2 * k + 1];
                                    ui cond_vertex_M = partial_subgraphs[WARPID][ID2order[cond_vertex]];
                                    if (cond == CondOperator::LESS_THAN)
                                    {
                                        if (cond_vertex_M <= vertex)
                                        {
                                            pred = false;
                                            break;
                                        }
                                    }
                                    else if (cond == CondOperator::LARGER_THAN)
                                    {
                                        if (cond_vertex_M >= vertex)
                                        {
                                            pred = false;
                                            break;
                                        }
                                    }
                                    else if (cond == CondOperator::NON_EQUAL)
                                    {
                                        if (cond_vertex_M == vertex)
                                        {
                                            pred = false;
                                            break;
                                        }
                                    }
                                }
                                if (pred)
                                {
                                    for (ui j = 0; j < bnCount; ++j)
                                    {
                                        ui u_prime = afterBackNeighbors[querySize[0] * id + j];
                                        ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                                        uintE u_prime_M_st = ctx->d_row_ptrs[u_prime_M];
                                        uintE u_prime_M_en = ctx->d_row_ptrs[u_prime_M + 1];
                                        pred = binarySearch(ctx->d_cols, u_prime_M_st, u_prime_M_en, vertex);
                                        if (!pred) break;
                                    }
                                    if (pred)
                                    {
                                        if (sglen + 1 == querySize[0])
                                        {
                                            // atomicAdd(&total_counts_[GLWARPID], 1);
                                            thread_count += 1;
                                        }
                                    }
                                }
                            }
                            if (sglen + 1 != querySize[0])
                            {
                                ui val = pred ? vertex : 0;
                                len = writeToTemp(val, 1, pred, len);
                            }
                        }

                        if (sglen + 1 == querySize[0])
                        {
                            continue;
                        }
                        else
                        {
                            if (NEXT_MODE == StoreStrategy::EXPAND)
                            {
                                auto vt = sg->append_batch(sglen + 1, len, StoreStrategy::EXPAND);
                                if (sg->isOverflow())
                                    return;
                                for (ui i = LANEID; i < len; i += 32)
                                {
                                    for (ui j = 0; j < sglen; ++j)
                                    {
                                        auto k = vt + i * (sglen + 1) + j;
                                        sg->wrBuff.vertices[k] = partial_subgraphs[WARPID][j];
                                    }
                                    sg->wrBuff.vertices[vt + i * (sglen + 1) + sglen] = tempv[i + GLWARPID * TEMPSIZE]; // add q on the back
                                } 
                            }
                            else if (NEXT_MODE == StoreStrategy::PREFIX)
                            {
                                for (ui batch_id = 0; batch_id < len; batch_id += BATCH_SIZE)
                                {
                                    ui min = len - batch_id < BATCH_SIZE ? len - batch_id : BATCH_SIZE;
                                    auto vt = sg->append_batch(sglen, min, StoreStrategy::PREFIX);
                                    if (sg->isOverflow())
                                        return;
                                    for (ui i = LANEID; i < sglen; i += 32)
                                    {
                                        auto k = vt + i;
                                        sg->wrBuff.vertices[k] = partial_subgraphs[WARPID][i];
                                    }
                                    for (ui i = LANEID; i < min; i += 32)
                                        sg->wrBuff.vertices[vt + sglen + i] = tempv[batch_id + i + GLWARPID * TEMPSIZE]; // add q on the back 
                                }
                            }
                        }
                    }
                } while (base_i < pu_en);
            }
            else
            {
                if (so.st + LANEID < so.en)
                    partial_subgraphs[WARPID][LANEID] = sg->rdBuff.vertices[so.st + LANEID];
                __syncwarp();
                
                // select the pivot with least # of candidates
                ui bnCount = backNeighborCount[id];
                ui u_prime = backNeighbors[querySize[0] * id];
                ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                uintE u_prime_M_st = ctx->d_row_ptrs[u_prime_M];
                uintE u_prime_M_en = ctx->d_row_ptrs[u_prime_M + 1];
                uintE min = u_prime_M_en - u_prime_M_st;
                ui parent_u = u_prime;
                for (ui i = 1; i < bnCount; ++i)
                {
                    ui u_prime = backNeighbors[querySize[0] * id + i];
                    ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                    uintE u_prime_M_st = ctx->d_row_ptrs[u_prime_M];
                    uintE u_prime_M_en = ctx->d_row_ptrs[u_prime_M + 1];
                    uintE neigh_len = u_prime_M_en - u_prime_M_st;
                    if (neigh_len < min)
                    {
                        min = neigh_len;
                        parent_u = u_prime;
                    }
                }

                ui parent_u_M = partial_subgraphs[WARPID][ID2order[parent_u]];

                uintE pu_st = ctx->d_row_ptrs[parent_u_M];
                uintE pu_en = ctx->d_row_ptrs[parent_u_M + 1];

                ui len = 0;
                bool pred;
                ui condCount = condNum[u];
                ui vertex;
                uintE base_i = pu_st;

                do {
                    len = 0;
                    for ( ; base_i < pu_en; base_i += 32)
                    {
                        uintE il = base_i + LANEID;
                        pred = il < pu_en;

                        if (pred)
                        {
                            vertex = ctx->d_cols[il];
                            for (ui k = 0; k < condCount; ++k)
                            {
                                ui cond = condOrder[u * querySize[0] * 2 + 2 * k];
                                ui cond_vertex = condOrder[u * querySize[0] * 2 + 2 * k + 1];
                                ui cond_vertex_M = partial_subgraphs[WARPID][ID2order[cond_vertex]];
                                if (cond == CondOperator::LESS_THAN)
                                {
                                    if (cond_vertex_M <= vertex)
                                    {
                                        pred = false;
                                        break;
                                    }
                                }
                                else if (cond == CondOperator::LARGER_THAN)
                                {
                                    if (cond_vertex_M >= vertex)
                                    {
                                        pred = false;
                                        break;
                                    }
                                }
                                else if (cond == CondOperator::NON_EQUAL)
                                {
                                    if (cond_vertex_M == vertex)
                                    {
                                        pred = false;
                                        break;
                                    }
                                }
                            }
                            if (pred)
                            {
                                for (ui j = 0; j < bnCount; ++j)
                                {
                                    ui u_prime = backNeighbors[querySize[0] * id + j];
                                    if (u_prime == parent_u)
                                        continue;
                                    ui u_prime_M = partial_subgraphs[WARPID][ID2order[u_prime]];
                                    uintE u_prime_M_st = ctx->d_row_ptrs[u_prime_M];
                                    uintE u_prime_M_en = ctx->d_row_ptrs[u_prime_M + 1];               
                                    pred = binarySearch(ctx->d_cols, u_prime_M_st, u_prime_M_en, vertex);
                                    if (!pred) break;
                                }
                                if (pred)
                                {
                                    if (sglen + 1 == querySize[0])
                                    {
                                        // atomicAdd(&total_counts_[GLWARPID], 1); // TODO: no good
                                        thread_count += 1;
                                    }
                                }
                            }
                        }
                        if (sglen + 1 != querySize[0])
                        {
                            ui val = pred ? vertex : 0;
                            len = writeToTemp(val, 1, pred, len);

                            if (len >= TEMPSIZE - 32) 
                            {
                                if (GTHID == 0) printf("# ");
                                base_i += 32;
                                break;
                            }
                        }
                    }

                    if (sglen + 1 == querySize[0])
                    {
                        // total_counts_[GLWARPID] += len;
                        continue;
                    }
                    else
                    {
                        if (NEXT_MODE == StoreStrategy::EXPAND)
                        {
                            auto vt = sg->append_batch(sglen + 1, len, StoreStrategy::EXPAND);
                            if (sg->isOverflow())
                                return;
                            for (ui i = LANEID; i < len; i += 32)
                            {
                                for (ui j = 0; j < sglen; ++j)
                                {
                                    auto k = vt + i * (sglen + 1) + j;
                                    sg->wrBuff.vertices[k] = partial_subgraphs[WARPID][j];
                                }
                                sg->wrBuff.vertices[vt + i * (sglen + 1) + sglen] = tempv[i + GLWARPID * TEMPSIZE]; // add q on the back
                            }
                        }
                        else if (NEXT_MODE == StoreStrategy::PREFIX)
                        {
                            for (ui batch_id = 0; batch_id < len; batch_id += BATCH_SIZE)
                            {
                                ui min = len - batch_id < BATCH_SIZE ? len - batch_id : BATCH_SIZE;
                                auto vt = sg->append_batch(sglen, min, StoreStrategy::PREFIX);
                                if (sg->isOverflow())
                                    return;
                                for (ui i = LANEID; i < sglen; i += 32)
                                {
                                    auto k = vt + i;
                                    sg->wrBuff.vertices[k] = partial_subgraphs[WARPID][i];
                                }
                                for (ui i = LANEID; i < min; i += 32)
                                    sg->wrBuff.vertices[vt + sglen + i] = tempv[batch_id + i + GLWARPID * TEMPSIZE]; // add q on the back 
                            }
                        }
                    }
                } while (base_i < pu_en);
            }
        }
        atomicAdd(&total_counts_[GLWARPID], thread_count);
    }


    __device__ void loadFromHost()
    {
        for (unsigned int i = GLWARPID; i < HOSTCHUNK; i += N_WARPS)
        {
            auto so = sgHost->rdBuff.next();
            if (sgHost->rdBuff.isEnd(so.st))
                break;
            unsigned int sglen = so.en - so.st;
            Index vt = sg->append(sglen);
            for (Index i = so.st + LANEID, j = vt + LANEID; i < so.en; i += 32, j += 32)
            {
                sg->wrBuff.vertices[j] = sgHost->rdBuff.vertices[i];
            }
        }
    }

    __device__ void dumpToHost(SubgraphOffsets *so)
    {
        if(sgHost->overflow[0]) {
            // host overflow occured, return from here and end the program by worker
            return;
        }
        ull vt = sgHost->append(so->en - so->st);

        for (ull i = vt + LANEID, j = so->st + LANEID; j < so->en; i += 32, j += 32)
        {
            sgHost->wrBuff.vertices[i] = sg->rdBuff.vertices[j];
        }
    }


    __device__ void expand()
    {
        // do nothing...
    }

    void completion()
    {
        // std::cout << "Matches: " << total[0] << std::endl;
    }

    void iterationFailed()
    {
        // do nothing...
    }
    void iterationSuccess()
    {
        // do nothing...
    }
};


#endif