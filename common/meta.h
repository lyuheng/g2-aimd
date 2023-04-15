
#ifndef COMMON_META_H
#define COMMON_META_H

#include <string>
#include <vector>
#include <assert.h>
#include <iostream>
#include <algorithm>


typedef unsigned int ui;
typedef unsigned long long int ull;

typedef unsigned int uintV;
typedef unsigned long long int uintE;


const static double kDeviceMemoryUnit = 16;  // P100
const static size_t kDeviceMemoryLimits[8] = {(size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024),
                                              (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024),
                                              (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024)};

enum CondOperator { LESS_THAN, LARGER_THAN, NON_EQUAL, OPERATOR_NONE };

enum StoreStrategy { EXPAND, PREFIX, COUNT };

enum ComputeStrategy { INTERSECTION, ENUMERATION };

#endif //COMMON_META_H