#ifndef DEVICE_CUDA_CONTEXT_H
#define DEVICE_CUDA_CONTEXT_H

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <vector>

#include "device/device_memory_info.h"

#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __forceinline__ __device__ __host__

#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(-1);
    }
}

template <typename T>
HOST_DEVICE void Swap(T &a, T &b)
{
    T tmp = a;
    a = b;
    b = tmp;
}
template <typename T>
HOST_DEVICE T Min(const T &a, const T &b)
{
    return (a < b) ? a : b;
}
template <typename T>
HOST_DEVICE T Max(const T &a, const T &b)
{
    return (a > b) ? a : b;
}


// ================== memory operations ========================
template <typename T>
void DToH(T *dest, const T *source, size_t count)
{
    CUDA_ERROR(cudaMemcpy(dest, source, count * sizeof(T), cudaMemcpyDeviceToHost));
}
template <typename T>
void DToD(T *dest, const T *source, size_t count)
{
    CUDA_ERROR(cudaMemcpy(dest, source, sizeof(T) * count, cudaMemcpyDeviceToDevice));
}
template <typename T>
void HToD(T *dest, const T *source, size_t count)
{
    CUDA_ERROR(cudaMemcpy(dest, source, sizeof(T) * count, cudaMemcpyHostToDevice));
}
template <typename T>
void DToH(T *dest, const T *source, size_t count, cudaStream_t stream)
{
    CUDA_ERROR(cudaMemcpyAsync(dest, source, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
}
template <typename T>
void DToD(T *dest, const T *source, size_t count, cudaStream_t stream)
{
    CUDA_ERROR(cudaMemcpyAsync(dest, source, sizeof(T) * count, cudaMemcpyDeviceToDevice, stream));
}
template <typename T>
void HToD(T *dest, const T *source, size_t count, cudaStream_t stream)
{
    CUDA_ERROR(cudaMemcpyAsync(dest, source, sizeof(T) * count, cudaMemcpyHostToDevice, stream));
}

template <typename T>
void DToH(std::vector<T> &dest, const T *source, size_t count)
{
    dest.resize(count);
    CUDA_ERROR(cudaMemcpy(dest.data(), source, sizeof(T) * count, cudaMemcpyDeviceToHost));
}
template <typename T>
void HToD(T *dest, const std::vector<T> &source, size_t count)
{
    CUDA_ERROR(cudaMemcpy(dest, source.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
}

template <typename T>
T GetD(T *source)
{
    T ret;
    DToH(&ret, source, 1);
    return ret;
}
template <typename T>
T GetD(T *source, cudaStream_t stream)
{
    T ret;
    DToH(&ret, source, 1, stream);
    return ret;
}
template <typename T>
void SetD(T *dest, T v)
{
    HToD(dest, &v, 1);
}
template <typename T>
void SetD(T *dest, T v, cudaStream_t stream)
{
    HToD(dest, &v, 1, stream);
}

class CudaContext
{
public:
    CudaContext(DeviceMemoryInfo *dev_mem, cudaStream_t stream) : dev_mem_(dev_mem), stream_(stream) {}
    
    cudaStream_t Stream() { return stream_; }

    ///// basic memory operation API //////
    void *Malloc(size_t size)
    {
        void *ret = SafeMalloc(size);
        // Memory statistics is updated after allocation because
        // the allocator needs to behave according to the current
        // available memory.
        dev_mem_->Consume(size);
        return ret;
    }

    void Free(void *p, size_t size)
    {
        SafeFree(p);
        dev_mem_->Release(size);
    }

    ///////// without tracking memory statistics /////
    // To support the case when the associated size of a pointer
    // cannot be abtained on Free, e.g., internal temporary memory
    // allocation in Thrust.
    // We should avoid use this API as much as possible.
    void *UnTrackMalloc(size_t size)
    {
        void *ret = SafeMalloc(size);
        return ret;
    }

    void UnTrackFree(void *p) { SafeFree(p); }


    ///////  info /////
    DeviceMemoryInfo *GetDeviceMemoryInfo() const { return dev_mem_; }

protected:
    ////// malloc and free implementation /////
    // the inherited class is recommended to
    // override them to modify the implementation.
    // By default we use SafeMalloc and SafeFree.

    void *DirectMalloc(size_t size)
    {
        void *ret = NULL;
        CUDA_ERROR(cudaMalloc(&ret, size));
        return ret;
    }

    void DirectFree(void *p) { CUDA_ERROR(cudaFree(p)); }

    void *SafeMalloc(size_t size)
    {
        if (dev_mem_->IsAvailable(size))
        {
            return DirectMalloc(size);
        }
        else
        {
            fprintf(stderr, "Insufficient device memory\n");
            void *ret = NULL;
            // allocate from unified memory
            CUDA_ERROR(cudaMallocManaged(&ret, size));
            CUDA_ERROR(cudaMemPrefetchAsync(ret, size, dev_mem_->GetDevId()));
            return ret;
        }
    }

    void SafeFree(void *p) { CUDA_ERROR(cudaFree(p)); }

protected:
    DeviceMemoryInfo *dev_mem_;
    cudaStream_t stream_;
};


#endif