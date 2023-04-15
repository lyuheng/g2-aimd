#ifndef SYSTEM_WORK_CONTEXT_H
#define SYSTEM_WORK_CONTEXT_H

#include <cuda_runtime.h>

#include "common/meta.h"
#include "device/cuda_context.h"

struct WorkContext
{
    size_t thread_num;

    ull ans; // answer

    CudaContext *context;

    // graph data
    size_t *sources_num; // change to satisfy GPU access
    uintV *sources;

    // only for squential executor
    uintE *row_ptrs;
    uintV *cols;

    // device graph data
    // DeviceArray<uintE> *d_row_ptrs; // not GPU accessible 
    // DeviceArray<uintV> *d_cols;

    uintE *d_row_ptrs;
    uintV *d_cols;

    ui *level;

    WorkContext()
    {
        thread_num = 1;
        ans = 0;
        context = nullptr;
        sources_num = 0;
        sources = nullptr;
        row_ptrs = nullptr;
        cols = nullptr;
        d_row_ptrs = nullptr;
        d_cols = nullptr;
    }

    ~WorkContext()
    {
        // context = nullptr;    
        // cudaFreeHost(sources);
        // cudaFreeHost(row_ptrs);
        // cudaFreeHost(cols);
        // delete d_row_ptrs;
        // delete d_cols;
    }
};

#endif