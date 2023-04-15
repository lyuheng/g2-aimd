#ifndef SYSTEM_APPBASE_H
#define SYSTEM_APPBASE_H

#include "common/graph.h"
#include "common/gpu_env.h"
#include "system/subgraph_container.h"
#include "system/work_context.h"
#include <cuda_runtime.h>


struct GMContext
{
    std::vector<uintV> matchOrderHost;
    std::vector<uintV> ID2orderHost;
    uintV *backNeighborCountHost;
    uintV *backNeighborsHost; 
    uintV *parentHost;
    bool *shareIntersectionHost;
    size_t sz;

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

    std::vector<StoreStrategy> strategyHost;
    ui *movingLvlHost;

    void AddGMContext(std::vector<uintV> &matchOrderHost_, std::vector<uintV> &ID2orderHost_, 
                        uintV* backNeighborCountHost_, uintV* backNeighborsHost_, uintV *parentHost_, 
                        size_t sz_, uintV *condOrderHost_, uintV *condNumHost_, bool *shareIntersectionHost_,
                        uintV *preBackNeighborCountHost_, uintV *preBackNeighborsHost_, uintV *preCondOrderHost_,
                        uintV *preCondNumHost_, uintV *afterBackNeighborCountHost_, uintV *afterBackNeighborsHost_,
                        uintV *afterCondOrderHost_, uintV *afterCondNumHost_, std::vector<StoreStrategy> &strategyHost_,
                        ui *movingLvlHost_)
    {
        backNeighborCountHost = backNeighborCountHost_;
        backNeighborsHost = backNeighborsHost_;
        parentHost = parentHost_;
        matchOrderHost = std::move(matchOrderHost_);
        ID2orderHost = std::move(ID2orderHost_);
        sz = sz_;
        condOrderHost = condOrderHost_;
        condNumHost = condNumHost_;
        shareIntersectionHost = shareIntersectionHost_;

        preBackNeighborCountHost = preBackNeighborCountHost_;
        preBackNeighborsHost = preBackNeighborsHost_;
        preCondOrderHost = preCondOrderHost_;
        preCondNumHost = preCondNumHost_;

        afterBackNeighborCountHost = afterBackNeighborCountHost_;
        afterBackNeighborsHost = afterBackNeighborsHost_;
        afterCondOrderHost = afterCondOrderHost_;
        afterCondNumHost = afterCondNumHost_;

        strategyHost = strategyHost_;
        movingLvlHost = movingLvlHost_;
    }
};


template <template <typename> class BuffType, class GraphType = Graph>
class AppBase
{
public:
    SubgraphContainer<BuffType<Index> > *sg;
    SubgraphContainer<BuffType<ull> > *sgHost;

    // GraphType *dp;

    // void setGraph(GraphType &g)
    // {
    //     chkerr(cudaMallocManaged(&dp, sizeof(GraphType)));
    //     dp->allocateMemory(g);
    // }

    WorkContext *ctx; // can be accessed on GPU
    ull *total_counts_; // can be accessed on GPU


    // NOT USED!!!!
    void setWorkContext(WorkContext *work_context) { ctx = work_context; }
    
    void initialize(size_t reserved_mem)
    {

        chkerr(cudaMallocManaged(&sg, sizeof(SubgraphContainer<BuffType<Index>>)));
        chkerr(cudaMallocManaged(&sgHost, sizeof(SubgraphContainer<BuffType<unsigned long long>>)));
        

        // sg reserves all the remaining memory on device. Hence it should be called after
        // all other memories are allocated.
        size_t sz = BuffType<Index>::sizeOf();
        size_t total, free;
        cudaMemGetInfo(&free, &total);

        // leave some memory for pointers and other variables...
        free -= 500'000'000 + reserved_mem;

        // sum of all the attributes size, multiply by 2, as there are two buffers...
        size_t alloc = free / (2 * sz);
 
        sg->allocateMemory(alloc);
        sgHost->allocateMemory();
    }

    __device__ virtual void generateSubgraphs(unsigned int base) = 0;
    __device__ virtual void loadFromHost() = 0;
    __device__ virtual void processSubgraphs() = 0;
    __device__ virtual void expand() = 0;

    virtual void allocateMemory() = 0;
    virtual void iterationFailed() = 0;
    virtual void iterationSuccess() = 0;
    virtual void completion() = 0;


    // =============== dedicated for subgraph matching =====================
    GMContext context;

    ui *matchOrder;
    ui *ID2order; 
    ui *backNeighborCount;
    ui *backNeighbors;
    ui *parent;
    bool *shareIntersection;

    ui *querySize;

    ui *condOrder;
    ui *condNum;

    uintV *preBackNeighborCount;
    uintV *preBackNeighbors;
    uintV *preCondOrder;
    uintV *preCondNum;

    uintV *afterBackNeighborCount;
    uintV *afterBackNeighbors;
    uintV *afterCondOrder;
    uintV *afterCondNum;

    StoreStrategy *strategy;
    ui *movingLvl;


    void allocateGMMemory()
    {
        chkerr(cudaMallocManaged((void **)&querySize, sizeof(ui)));
        querySize[0] = context.sz;

        chkerr(cudaMalloc((void **)&matchOrder, context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(matchOrder, context.matchOrderHost.data(), context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&ID2order, context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(ID2order, context.ID2orderHost.data(), context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&backNeighborCount, context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(backNeighborCount, context.backNeighborCountHost, context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&backNeighbors, context.sz*context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(backNeighbors, context.backNeighborsHost, context.sz*context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&parent, context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(parent, context.parentHost, context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&condOrder, 2*context.sz*context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(condOrder, context.condOrderHost, 2*context.sz*context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&condNum, context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(condNum, context.condNumHost, context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&shareIntersection, context.sz*sizeof(bool)));
        chkerr(cudaMemcpy(shareIntersection, context.shareIntersectionHost, context.sz*sizeof(bool), cudaMemcpyHostToDevice));



        chkerr(cudaMalloc((void **)&preBackNeighborCount, context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(preBackNeighborCount, context.preBackNeighborCountHost, context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&preBackNeighbors, context.sz*context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(preBackNeighbors, context.preBackNeighborsHost, context.sz*context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&preCondOrder, 2*context.sz*context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(preCondOrder, context.preCondOrderHost, 2*context.sz*context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&preCondNum, context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(preCondNum, context.preCondNumHost, context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterBackNeighborCount, context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(afterBackNeighborCount, context.afterBackNeighborCountHost, context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterBackNeighbors, context.sz*context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(afterBackNeighbors, context.afterBackNeighborsHost, context.sz*context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterCondOrder, 2*context.sz*context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(afterCondOrder, context.afterCondOrderHost, 2*context.sz*context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&afterCondNum, context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(afterCondNum, context.afterCondNumHost, context.sz*sizeof(ui), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&strategy, (context.sz + 1)*sizeof(StoreStrategy)));
        chkerr(cudaMemcpy(strategy, context.strategyHost.data(), (context.sz + 1)*sizeof(StoreStrategy), cudaMemcpyHostToDevice));

        chkerr(cudaMalloc((void **)&movingLvl, context.sz*sizeof(ui)));
        chkerr(cudaMemcpy(movingLvl, context.movingLvlHost, context.sz*sizeof(ui), cudaMemcpyHostToDevice));
    }

    // =============== dedicated for subgraph matching =====================
};

#endif