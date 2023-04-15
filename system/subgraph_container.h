#ifndef SYSTEM_SUBGRPAH_CONTAINER_H
#define SYSTEM_SUBGRPAH_CONTAINER_H

#include "common/meta.h"
#include "common/gpu_env.h"
#include "system/buffer.h"
#include <cuda_runtime.h>

/**
 * @brief This is a templated subgraphs container.
 *
 * It uses two buffers: wrBuff, rdBuff to write and process subgraphs respectively.
 * The required template class is a buffer class, where user can specify data required for each subgraph
 *
 */
template <class BuffType>
class SubgraphContainer
{
public:
    /**
     * @brief Writing buffer of the container
     * This buffer will be used to generate new subgraphs
     */
    BuffType wrBuff;
    // C2: writing buffer, subgraphs are always written on C2.

    /**
     * @brief Reading buffer of the container
     * The subgraphs placed on this buffer will be used to process.
     *
     */
    BuffType rdBuff;

    /**
     * @brief This buffer is used to store subgraphs on host memory
     *
     */
    // BuffType hostBuff;

    size_t *buffsize;

    volatile bool *overflow; 
    unsigned int *chunk;
    unsigned int *head;
    unsigned int step, successCount = 0, addStep, threshold;
    bool recover = true;

    // ================= add by lyuheng ===============

    ui *warpAssigned;

    int *warpCount;
    int *warpCountMoving;
    // ================================================

    /**
     * @brief It allocates memory to device buffers and container internal variables.
     *
     */

    void allocateMemory(ull sz)
    {
        chkerr(cudaMallocManaged((void **)&buffsize, sizeof(ull)));
        chkerr(cudaMallocManaged((void **)&head, sizeof(ull)));
        buffsize[0] = sz;
        rdBuff.allocateMemory(buffsize[0]);
        wrBuff.allocateMemory(buffsize[0]);

        // this version allocates on host memory
        // hostBuff.allocateMemory();

        chkerr(cudaMallocManaged((void **)&overflow, sizeof(bool)));
        chkerr(cudaMallocManaged((void **)&chunk, sizeof(int)));
        overflow[0] = false;
        chunk[0] = MAXCHUNK;
        recover = true;
        successCount = 0;

        // chkerr(cudaMallocManaged((void **)&warpAssigned, sizeof(ui)));
        // chkerr(cudaMallocManaged((void **)&warpCount, (N_WARPS+1) * sizeof(int)));
        // chkerr(cudaMallocManaged((void **)&warpCountMoving, (N_WARPS+1) * sizeof(int)));
    }
    /**
     * @brief Allocated Memory to host. This version is called for Ho
     * 
     */
    void allocateMemory()
    {

        chkerr(cudaMallocManaged((void **)&buffsize, sizeof(ull)));
        chkerr(cudaMallocManaged((void **)&head, sizeof(ull)));


        rdBuff.allocateMemory();
        wrBuff.allocateMemory();

        // this version allocates on host memory
        // hostBuff.allocateMemory();

        chkerr(cudaMallocManaged((void **)&overflow, sizeof(bool)));
        chkerr(cudaMallocManaged((void **)&chunk, sizeof(int)));
        overflow[0] = false;
        chunk[0] = HOSTCHUNK;

        // chkerr(cudaMallocManaged((void **)&warpAssigned, sizeof(ui)));
        // chkerr(cudaMallocManaged((void **)&warpCount, (N_WARPS+1) * sizeof(int)));
        // chkerr(cudaMallocManaged((void **)&warpCountMoving, (N_WARPS+1) * sizeof(int)));
    }
    /**
     * @brief Swaps the reading and writing buffers
     *
     * This function is called after completing every iteration,
     * so that new spawned subgraphs are processed in next iteration
     *
     */
    void swapBuffers()
    {
        // using default assignment operator for shallow copy...
        BuffType temp;
        temp = rdBuff;
        rdBuff = wrBuff;
        wrBuff = temp;

        // reset the read head, and offset tails gets
        // the value how much have been written
        wrBuff.clear();
        rdBuff.ohead[0] = 0;
        if(overflow[0]) {
            recover = false;
        }
        overflow[0] = false;
    }

    void assign()
    {
        // assign tasks (subgraphs) to each warp, # subgraphs
        warpAssigned[0] = (rdBuff.otail[0] / 2 ) / N_WARPS;
        ui remain = (rdBuff.otail[0] / 2 ) % N_WARPS;
        ui counter = 0;
        for (ui i = 0; i < remain; ++i)
        {   
            warpCount[i] = counter;
            warpCountMoving[i] = counter;
            counter += warpAssigned[0] + 1;
        }
        for (ui i = remain; i < N_WARPS; ++i)  
        {
            warpCount[i] = counter;
            warpCountMoving[i] = counter;
            counter += warpAssigned[0];
        }
        warpCount[N_WARPS] = rdBuff.otail[0] / 2;
        warpCountMoving[N_WARPS] = rdBuff.otail[0] / 2;

        // std::cout << "warpAssigned = " <<  warpAssigned[0] << std::endl;
    }
    /**
     * @brief Checks the status of container, can be used to decide if all subgraphs have been processed
     * in the current iteration.
     *
     * @return true if container is empty
     */
    bool isEmpty()
    {
        return rdBuff.isEmpty();
    }

    /**
     * @brief Buffers can overflow when spawning subgraphs
     *
     * @return true if buffer limit exceeded
     */
    __device__ __host__ bool isOverflow()
    {
        return overflow[0] && chunk[0] > MINCHUNK;
    }

    void adjustChunk()
    {
        // Reducing Chunk 
        if(overflow[0] && chunk[0]>MINCHUNK){
            threshold = max(MINTHRESHOLD, chunk[0]/2);
            chunk[0] = max(chunk[0] / DECFACTOR, MINCHUNK); // chunk size is reduced by factor
            step = max((int)(chunk[0] * ADDPC), MINSTEP); // additive increment
            addStep = step;
            std::cout<<"--- chunk: "<<chunk[0];
            successCount = 0;
        }
        // Increasing Chunk
        // recover is false if flag was raised in any iteration of internal loop i.e. processing C1 and swapping C1, C2
        else if(recover && chunk[0]<MAXCHUNK){
            // if(successCount==SUCCESS_ITER){
            //     step*=2;
            //     successCount = 0;
            // }
            // successCount++;
            if(chunk[0]<threshold)
                chunk[0] = min(chunk[0] + step, MAXCHUNK);
            else
                chunk[0] = min(chunk[0]+addStep, MAXCHUNK);
            step*=2;
            std::cout<<"+++ chunk: "<<chunk[0];
        }
        else
            std::cout<<"... chunk: "<<chunk[0];
        // reset tail pointers... 
        recover = true;
        // hostBuff.clear();
        rdBuff.clear();
        wrBuff.clear();
    }

    __device__ __host__ bool isOverflowToHost()
    {
        return overflow[0] && chunk[0] <= MINCHUNK;
    }
    /**
     * @brief It reserves a space for a new subgraph of @param sglen size. A warp can call this function to reserve.
     * The function only reserves a space in the container, using the return value index the graph
     * should be written afterwards.
     * @param sglen length of subgraph to be pushed on the container.
     * @return starting index where graph can be written.
     */
    __device__ ull append(Index sglen)
    {
        return wrBuff.append(sglen, overflow);
    }
    __device__ ull append(Index sglen, Index midpos)
    {
        return wrBuff.append(sglen, midpos, overflow);
    }

    __device__ ull append_batch(Index sglen, int num, StoreStrategy mode)
    {
        return wrBuff.append_batch(sglen, num, overflow, mode);
    }

     __device__ ull append_thread(Index sglen)
    {
        return wrBuff.append_thread(sglen, overflow);
    }

    __device__ SubgraphOffsets next()
    {
        return rdBuff.next();
    }

    /**
     * @brief Removes a subgraph in FIFO order
     * Calling next on empty subgraph returns the end() indices.
     * @return returns {st, en} indices of the subgraph.
     */
    __device__ SubgraphOffsets next(StoreStrategy mode)
    {
        return rdBuff.next(mode);
    }
    /**
     * @brief returns last indices of the container's reading buffer.
     * Can be used to decide if all subgraphs have been processed.
     *
     * @return last invalid index of the reading buffer.
     */
    __device__ bool isEnd(SubgraphOffsets so)
    {
        return rdBuff.isEnd(so);
        
        // TODO: remove after debugging
        // return warpCountMoving[GLWARPID] == warpCount[GLWARPID + 1];
    }

    __device__ bool isEnd(unsigned long long st)
    {
        return rdBuff.isEnd(st);
        
        // TODO: remove after debugging
        // return warpCountMoving[GLWARPID] == warpCount[GLWARPID + 1];
    }
};


#endif