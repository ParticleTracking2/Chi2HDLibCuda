//============================================================================
// Name        : Chi2HD_CudaFFT.cu
// Author      : Juan Silva
// Version     :
// Copyright   : All rights reserved
// Description : Funciones para trabajar el algoritmo de minimos cuadrados. Parte FFT
//============================================================================


#include "Headers/Chi2HD_CudaUtils.h"

#if defined(__cplusplus)
extern "C" {
#endif

/******************
 * Utilidades
 ******************/
/**
 * Numeros de bloques a ejecutar.
 */
unsigned int _findOptimalGridSize(cuMyArray2D *arr){
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	unsigned int maxThreads = deviceProp.maxThreadsPerBlock;
	unsigned int requiredGrid = ceil(arr->getSize()/maxThreads)+1;
	if(requiredGrid < deviceProp.maxGridSize[0])
		return requiredGrid;
	return deviceProp.maxGridSize[0];
}

/**
 * Cantidad de threads dentro de cada bloque.
 */
unsigned int _findOptimalBlockSize(cuMyArray2D *arr){
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	return deviceProp.maxThreadsPerBlock;
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
