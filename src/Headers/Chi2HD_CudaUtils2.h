/*
 * Chi2HD_CudaUtils.h
 *
 *  Created on: 23/11/2011
 *      Author: juanin
 */

#ifndef CHI2HD_CUDAUTILS2_H_
#define CHI2HD_CUDAUTILS2_H_

#include <stdlib.h>
#include <stdio.h>
#include <utility>
#include "Chi2HD_CudaStructs.h"

#if defined(__cplusplus)
extern "C++" {
#endif

unsigned int _findOptimalGridSize(unsigned int size);

std::pair<unsigned int, unsigned int> _findOptimalGridSize(unsigned int sizeX, unsigned int sizeY);

unsigned int _findOptimalBlockSize(unsigned int size);

std::pair<unsigned int, unsigned int> _findOptimalBlockSize(unsigned int sizeX, unsigned int sizeY);

void manageError(cudaError_t err);

#if defined(__cplusplus)
}
#endif

#endif
