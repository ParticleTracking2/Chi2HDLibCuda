//============================================================================
// Name        : Chi2LibcuUtils.cu
// Author      : Juan Silva
// Version     :
// Copyright   : All rights reserved
// Description : Utilidades para las llamadas a Kernel y errores generales en
//				 Cuda
//============================================================================


#include "Headers/Chi2LibcuUtils.h"

#if defined(__cplusplus)
extern "C++" {
#endif

/******************
 * Utilidades
 ******************/
/**
 * Numeros de bloques a ejecutar.
 */
unsigned int _findOptimalGridSize(unsigned int size){
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	unsigned int maxThreads = deviceProp.maxThreadsPerBlock;
	unsigned int requiredGrid = ceil(size/maxThreads)+1;
	if(requiredGrid < deviceProp.maxGridSize[0])
		return requiredGrid;
	return deviceProp.maxGridSize[0];
}

std::pair<unsigned int, unsigned int> _findOptimalGridSize(unsigned int sizeX, unsigned int sizeY){
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	unsigned int maxThreads = deviceProp.maxThreadsPerBlock;

	std::pair<unsigned int, unsigned int> ret;
	unsigned int requiredGridX = ceil(sizeX/maxThreads)+1;
	if(requiredGridX < deviceProp.maxGridSize[0])
		ret.first = requiredGridX;
	else
		ret.first = deviceProp.maxGridSize[0];

	unsigned int requiredGridY = ceil(sizeY/maxThreads)+1;
	if(requiredGridY < deviceProp.maxGridSize[1])
		ret.second = requiredGridY;
	else
		ret.first = deviceProp.maxGridSize[1];

	return ret;
}

/**
 * Cantidad de threads dentro de cada bloque.
 */
unsigned int _findOptimalBlockSize(unsigned int size){
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	return deviceProp.maxThreadsPerBlock;
}

std::pair<unsigned int, unsigned int> _findOptimalBlockSize(unsigned int sizeX, unsigned int sizeY){
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	std::pair<unsigned int, unsigned int> ret;
	ret.first = deviceProp.maxThreadsPerBlock;
	ret.second = deviceProp.maxThreadsPerBlock;

	return ret;
}

/**
 * Maneja los errores de CUDA
 */
void manageError(cudaError_t err){
	if(err != cudaSuccess){
		printf("CHI2HD_CUDA Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

#if defined(__cplusplus)
}
#endif
