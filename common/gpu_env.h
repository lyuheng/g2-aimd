#ifndef COMMON_GPU_ENV_H
#define COMMON_GPU_ENV_H

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <bits/stdc++.h>
#include "omp.h"

// below are from original file

// P100
// #define BLK_NUMS 56
// #define BLK_DIM 1024
// // A100
#define BLK_NUMS 108 
#define BLK_DIM 1024

#define WARPS_EACH_BLK (BLK_DIM >> 5)
#define MAX_NV 10000
#define N_THREADS (BLK_DIM * BLK_NUMS)
#define N_WARPS (BLK_NUMS * WARPS_EACH_BLK)
#define GLBUFFER_SIZE 1000000
#define THID threadIdx.x
#define WARP_SIZE 32
#define UINT unsigned int
#define DS_LOC string("")
#define OUTPUT_LOC string("../../output/")
#define REP 1
#define WARPID (THID >> 5)
#define LANEID (THID & 31)
#define BLKID blockIdx.x
#define FULL 0xFFFFFFFF
#define GLWARPID (BLKID * WARPS_EACH_BLK + WARPID)
#define GTHID (BLKID * N_THREADS + THID)
// #define GLWARPID (blockIdx.x*(BLK_DIM/32)+(threadIdx.x>>5))

// ****************BK specific parameters********************
// CHUNK is how many subproblems processes initially from the degeneracy ordered pool
// chunk value is reduced to half everytime a memory flow occurs.
// It can't reduce once reached to MINCHUNK
// #define MAXCHUNK 10'000'000
#define MAXCHUNK 1'000'000
#define MINCHUNK 1
#define MINSTEP 1

#define HOSTCHUNK 1'000'000 // chunk size for host memory buffer

// NSUBS is size of sg.offsets array, since every subgraph requires 2 items,
// hence number of supported subgraphs are half of this number
// #define NSUBS 8e8
// BUFFSIZE is length of sg.vertices and sg.labels
// #define BUFFSIZE 8e8
// tempsize is max size of a subgraph stored in temp area, This size is per warp
// in general it should be the size of max degree of supported graph
#define TEMPSIZE 100'000

#define HOST_BUFF_SZ 20'000'000'000ULL
#define HOST_OFFSET_SZ 1'000'000'000ULL

// #define HOST_BUFF_SZ 20
// #define HOST_OFFSET_SZ 1

// Reduction and Increment factors
#define DECFACTOR 8
#define INCFACTOR 2
#define THRESH 0
#define ADDPC 0.05
#define SUCCESS_ITER 0
#define MINTHRESHOLD 2000

#define R 'r'
#define P 'p'
#define X 'x'
#define Q 'q'

#define DEV __device__
#define DEVHOST __device__ __host__

typedef unsigned int Index;
typedef unsigned int VertexID;
typedef char Label;

#define DEGREESORT

inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout << cudaGetErrorString(code) << std::endl;
        exit(-1);
    }
}

void deviceQuery()
{
    cudaDeviceProp prop;
    int nDevices = 0, i;
    cudaError_t ierr;

    ierr = cudaGetDeviceCount(&nDevices);
    if (ierr != cudaSuccess)
    {
        printf("Sync error: %s\n", cudaGetErrorString(ierr));
    }

    for (i = 0; i < nDevices; ++i)
    {
        ierr = cudaGetDeviceProperties(&prop, i);
        printf("Device number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Compute capability: %d.%d\n\n", prop.major, prop.minor);

        printf("  Clock Rate: %d kHz\n", prop.clockRate);
        printf("  Total SMs: %d \n", prop.multiProcessorCount);
        printf("  Shared Memory Per SM: %lu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("  Registers Per SM: %d 32-bit\n", prop.regsPerMultiprocessor);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  L2 Cache Size: %d bytes\n", prop.l2CacheSize);
        printf("  Total Global Memory: %lu bytes\n", prop.totalGlobalMem);
        printf("  Memory Clock Rate: %d kHz\n\n", prop.memoryClockRate);

        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]);
        printf("  Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]);
        printf("  Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]);

        printf("  Max blocks in X-dimension of grid: %d\n", prop.maxGridSize[0]);
        printf("  Max blocks in Y-dimension of grid: %d\n", prop.maxGridSize[1]);
        printf("  Max blocks in Z-dimension of grid: %d\n\n", prop.maxGridSize[2]);

        printf("  Shared Memory Per Block: %lu bytes\n", prop.sharedMemPerBlock);
        printf("  Registers Per Block: %d 32-bit\n", prop.regsPerBlock);
        printf("  Warp size: %d\n\n", prop.warpSize);
    }
}

#endif