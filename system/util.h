#ifndef SYSTEM_UTIL_H
#define SYSTEM_UTIL_H

#include "common/meta.h"

// FIXME: PROBLEM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
__device__ bool binarySearch(unsigned int* arr, int low, int high, unsigned int val){
    int mid;
    high--; // here in this search high must be inclusive, otherwise it can overflow
    while(low<=high){
        mid = (high + low)/2;
        if(val == arr[mid]) return true;
        else if(val<arr[mid]) high = mid-1;
        else low = mid+1;
    }
    return false;
}

__device__ bool binarySearch(uintV* arr, uintE low, uintE high, ui val){
    uintE mid;
    if (high != 0) high--; // here in this search high must be inclusive, otherwise it can overflow
    
    while (low < high) {
        mid = low + (high - low)/2;   
        if (val == arr[mid]) return true;
        else if (arr[mid] < val) low = mid + 1;
        else high = mid;
    }
    return arr[low] == val;
}


__device__ bool linearSearch(unsigned int* data, unsigned int st, unsigned int en, unsigned int v)
{
    bool pred;
    unsigned int laneid = LANEID;
    unsigned int res;
    for(unsigned int k; st<en; st+=32){
        if(data[st] > v) return false; // this exploit the sorted nature of data, and can break early.
        k = st+laneid;
        pred = k < en && (v == data[k]);
        res = __ballot_sync(FULL, pred);
        if(res!=0) return true;
    }
    return false;
}

// returns index to write after scanning a warp
__device__ unsigned int scanIndex(bool pred)
{
    unsigned int bits = __ballot_sync(FULL, pred);
    unsigned int mask = FULL >> (31 - LANEID);
    unsigned int index = __popc(mask & bits) - pred; // to get exclusive sum subtract pred
    return index;
}

#endif