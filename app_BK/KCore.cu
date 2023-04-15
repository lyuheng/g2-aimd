#include "KCore.h"
#define VWARPSIZE 32
#define VWARPID (THID >> 5)
#define VWARPS_EACH_BLK (BLK_DIM >> 5)
#define VLANEID (THID & 31)
__global__ void selectNodesAtLevel(KCore kc)
{
    kc.selectNodesAtLevel();
}

__global__ void processNodes(KCore kc)
{
    kc.processNodes();
}

void KCore::allocateMemory(Graph &g)
{
    chkerr(cudaMallocManaged(&count, sizeof(Index)));
    chkerr(cudaMallocManaged(&degOrder, g.V * sizeof(Index)));
    chkerr(cudaMallocManaged(&dp, sizeof(Graph)));
    chkerr(cudaMallocManaged(&buff, sizeof(KCoreBuffer)));
    dp->allocateMemory(g);
    buff->allocateMemory();
    level = 0;
    count[0] = 0;
}
__device__ void KCore::selectNodesAtLevel()
{
    __shared__ unsigned int bufTail;
    if (THID == 0)
    {
        bufTail = 0;
    }
    __syncthreads();
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int base = 0; base < dp->V; base += N_THREADS)
    {
        unsigned int v = base + global_threadIdx;
        if (v >= dp->V)
            break;
        if (dp->degrees[v] == level)
        {
            unsigned int loc = atomicAdd(&bufTail, 1);
            buff->write(loc, v);
        }
    }
    __syncthreads();
    if (THID == 0)
    {
        buff->bufTails[blockIdx.x] = bufTail;
    }
}

__device__ void KCore::processNodes()
{
    __shared__ unsigned int bufTail;
    __shared__ unsigned int base;
    unsigned int regTail;
    unsigned int i;
    if (THID == 0)
    {
        bufTail = buff->bufTails[blockIdx.x];
        base = 0;
    }

    // bufTail is being incrmented within the loop,
    // warps should process all the nodes added during the execution of loop

    // for(unsigned int i = warp_id; i<bufTail ; i +=warps_each_block ){
    // this for loop is a wrong choice, as many threads will exit from the loop checking the condition
    while (true)
    {
        __syncthreads(); // syncthreads must be executed by all the threads
        if (base == bufTail)
            break; // all the threads will evaluate to true at same iteration
        // i = base + WARPID;
        i = base + VWARPID;
        regTail = bufTail;
        __syncthreads();

        if (i >= regTail)
            continue; // this warp won't have to do anything

        if (THID == 0)
        {
            base += min(VWARPS_EACH_BLK, regTail - base);
        }
        // bufTail is incremented in the code below:

        VertexID v = buff->read(i);
        Index start = dp->neighbors_offset[v];
        Index end = dp->neighbors_offset[v + 1];

        while (true)
        {
            __syncwarp();

            if (start >= end)
                break;

            unsigned int j = start + VLANEID;
            start += VWARPSIZE;
            if (j >= end)
                continue;

            unsigned int u = dp->neighbors[j];
            if (dp->degrees[u] > level)
            {

                unsigned int a = atomicSub(dp->degrees + u, 1);

                if (a == level + 1)
                {
                    unsigned int loc = atomicAdd(&bufTail, 1);
                    buff->write(loc, u);
                }

                if (a <= level)
                {
                    // node degree became less than the level after decrementing...
                    atomicAdd(dp->degrees + u, 1);
                }
            }
        }
    }

    if (bufTail > 0)
    {
        if (THID == 0)
            base = atomicAdd(count, bufTail); // atomic since contention among blocks
        __syncthreads();
        // Store degeneracy order...
        for (int i = THID; i < bufTail; i += BLK_DIM)
        {
#ifdef DEGREESORT
            degOrder[base + i] = buff->read(i); // needs to process it again if done this way
#else
            degOrder[buff->read(i)] = base + i;
            // printf("%d->%d ", buff->read(i), base+i);
#endif
        }
    }
}

Index *Degeneracy::degenerate()
{

    KCore kc;
    kc.allocateMemory(g);

    cout << "K-core Computation Started" << endl;

    auto tick = chrono::steady_clock::now();
    while (kc.count[0] < g.V)
    {
        kc.buff->resetTails();
        selectNodesAtLevel<<<BLK_NUMS, BLK_DIM>>>(kc);
        processNodes<<<BLK_NUMS, BLK_DIM>>>(kc);
        cudaDeviceSynchronize();
        // cout<<"*********Completed level: "<<kc.level<<", global_count: "<<kc.count[0]<<" *********"<<endl;
        kc.level++;
        degOrderOffsets.push_back(kc.count[0]);
    }
    cout << "Kcore Computation Done" << endl;
    cout << "KMax: " << kc.level - 1 << endl;
    cout << "Kcore-class execution Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - tick).count() << endl;
    chkerr(cudaMemcpy(degOrder, kc.degOrder, sizeof(unsigned int) * g.V, cudaMemcpyDeviceToHost));

    return degOrder;
}

void Degeneracy::degreeSort()
{
    ////////////////// degrees sorting after degenracy...
    auto tick = chrono::steady_clock::now();

    // sort each k-shell vertices based on their degrees.
    auto degComp = [&](auto a, auto b){
        return g.degrees[a]<g.degrees[b];
    };

    for (int i = 0; i < degOrderOffsets.size() - 1; i++)
        std::sort(degOrder + degOrderOffsets[i], degOrder+degOrderOffsets[i+1], degComp);


    VertexID* revOrder=new VertexID[g.V];
    // copy back the sorted vertices to rec array...
    for (int i = 0; i < g.V; i++)
        revOrder[degOrder[i]] = i;
    std::swap(degOrder, revOrder);

    delete[] revOrder;
    cout << "Degree Sorting Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - tick).count() << endl;
}

// Graph Degeneracy::degreeSortRecode()
// {
//     auto tick = chrono::steady_clock::now();
//     degreeSort();
//     Graph gRec =  recode();
//     Graph gRec;
//     gRec.degrees = new unsigned int[g.V];
//     gRec.neighbors = new unsigned int[g.E];
//     gRec.neighbors_offset = new unsigned int[g.V + 1];
//     gRec.V = g.V;
//     gRec.E = g.E;
//     cout << "Degrees copied" << endl;
//     for (int i = 0; i < g.V; i++)
//         gRec.degrees[i] = g.degrees[degOrder[i]];
//     map<unsigned int, unsigned int> recMapping;
//     for (int i = 0; i < g.V; i++)
//         recMapping[degOrder[i]] = i;

//     gRec.neighbors_offset[0] = 0;
//     std::partial_sum(gRec.degrees, gRec.degrees + g.V, gRec.neighbors_offset + 1);

//     for (int v = 0; v < g.V; v++)
//     {
//         unsigned int recv = degOrder[v];
//         unsigned int start = gRec.neighbors_offset[v];
//         unsigned int end = gRec.neighbors_offset[v + 1];
//         for (int j = g.neighbors_offset[recv], k = start; j < g.neighbors_offset[recv + 1]; j++, k++)
//         {
//             gRec.neighbors[k] = recMapping[g.neighbors[j]];
//         }
//         std::sort(gRec.neighbors + start, gRec.neighbors + end);
//     }
//     cout << "Degree Sorted Reordering Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - tick).count() << endl;
//     return gRec;
// }

Graph Degeneracy::recode()
{
    Graph gRec;
    gRec.degrees = new unsigned int[g.V];
    gRec.neighbors = new unsigned int[g.E];
    gRec.neighbors_offset = new unsigned int[g.V + 1];
    gRec.V = g.V;
    gRec.E = g.E;

    auto tick = chrono::steady_clock::now();
    cout << "Degrees copied" << endl;
    for (int i = 0; i < g.V; i++)
    {
        gRec.degrees[degOrder[i]] = g.degrees[i];
    }
    // for (int i = 0; i < g.V; i++){
    //     cout<<i<<" : "<<g.degrees[i]<<endl;
    // }
    // cout<<"---->"<<endl;
    // for (int i = 0; i < g.V; i++){
    //     cout<<i<<" : "<<gRec.degrees[i]<<endl;
    // }

    gRec.neighbors_offset[0] = 0;
    std::partial_sum(gRec.degrees, gRec.degrees + g.V, gRec.neighbors_offset + 1);

    for (int v = 0; v < g.V; v++)
    {
        unsigned int recv = degOrder[v];
        unsigned int start = gRec.neighbors_offset[recv];
        unsigned int end = gRec.neighbors_offset[recv + 1];
        for (int j = g.neighbors_offset[v], k = start; j < g.neighbors_offset[v + 1]; j++, k++)
        {
            gRec.neighbors[k] = degOrder[g.neighbors[j]];
        }
        std::sort(gRec.neighbors + start, gRec.neighbors + end);
    }
    cout << "Reordering Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - tick).count() << endl;

    return gRec;
}