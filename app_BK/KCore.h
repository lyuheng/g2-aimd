#ifndef KCORE_H
#define KCORE_H

#include "graph.h"

// #define DEGREESORT

class KCoreBuffer
{
public:
    Index *glBuffer;
    Index *bufTails;

    void allocateMemory()
    {
        chkerr(cudaMalloc(&glBuffer, sizeof(Index) * BLK_NUMS * GLBUFFER_SIZE));
        chkerr(cudaMalloc(&bufTails, sizeof(Index) * BLK_NUMS));
    }

    void resetTails()
    {
        cudaMemset(bufTails, 0, sizeof(unsigned int) * BLK_NUMS);
    }

    __device__ void write(unsigned int loc, unsigned int v)
    {
        assert(loc < GLBUFFER_SIZE);
        glBuffer[blockIdx.x * GLBUFFER_SIZE + loc] = v;
    }

    __device__ unsigned int read(unsigned int loc)
    {
        assert(loc < GLBUFFER_SIZE);
        return glBuffer[blockIdx.x * GLBUFFER_SIZE + loc];
    }
};
class KCore
{
public:
    Index *count;
    Index *degOrder;
    KCoreBuffer *buff;
    Graph *dp;
    unsigned int level;

    void allocateMemory(Graph &g);

    __device__ void selectNodesAtLevel();

    __device__ void processNodes();
};

class Degeneracy
{
    Graph g;
    Index *degOrder;
    vector<Index> degOrderOffsets;

public:
    Degeneracy(Graph &dg) : g(dg)
    {
        degOrder = new Index[g.V];
        degOrderOffsets.push_back(0);
    }
    void degreeSort();
    Index *degenerate();
    Graph recode();
};
#endif