/*
 * Chi2HD_CudaUtils.h
 *
 *  Created on: 23/11/2011
 *      Author: juanin
 */

#ifndef CHI2HD_CUDAUTILS_H_
#define CHI2HD_CUDAUTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include "Chi2HD_CudaStructs.h"

#if defined(__cplusplus)
extern "C" {
#endif

unsigned int _findOptimalGridSize(cuMyArray2D *arr);

unsigned int _findOptimalBlockSize(cuMyArray2D *arr);

void manageError(cudaError_t err);

#if defined(__cplusplus)
}
#endif

#endif
