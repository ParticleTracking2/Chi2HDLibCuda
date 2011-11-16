//============================================================================
// Name        : Chi2HD_Cuda.cu
// Author      : Juan Silva
// Version     :
// Copyright   : All rights reserved
// Description : Hello World in CUDA
//============================================================================

#include "Headers/Chi2HD_Cuda.h"

/**
Establece el dispositivo con mayor poder de computo.

int num_devices, device;
cudaGetDeviceCount(&num_devices);
if (num_devices > 1) {
      int max_multiprocessors = 0, max_device = 0;
      for (device = 0; device < num_devices; device++) {
              cudaDeviceProp properties;
              cudaGetDeviceProperties(&properties, device);
              if (max_multiprocessors < properties.multiProcessorCount) {
                      max_multiprocessors = properties.multiProcessorCount;
                      max_device = device;
              }
      }
      cudaSetDevice(max_device);
}
 */

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Maximos elementos para le GPU GTX590
 */
const unsigned int STD_THREAD_SIZE = 512;
const unsigned int MAX_THREAD_SIZE = 1024;

const unsigned int STD_BLOCK_SIZE = 512;
const unsigned int MAX_BLOCK_SIZE = 1024;

const unsigned int STD_GRID_SIZE = 1024;
const unsigned int MAX_GRID_SIZE = 65535;

unsigned int _findOptimalGridSize(cuMyArray2D *arr){
	return STD_GRID_SIZE;
}
unsigned int _findOptimalBlockSize(cuMyArray2D *arr){
	return STD_BLOCK_SIZE;
}

cuMyArray2D CHI2HD_createArray(unsigned int sx, unsigned int sy){
	cuMyArray2D ret;
	cudaError_t err = cudaMalloc((void**)&ret._array, (size_t) sx*sy*sizeof(float));

	if(err != cudaSuccess)
		exit(1);

	ret._host_array = 0;
	ret._device = 0;
	ret._sizeX = sx;
	ret._sizeY = sy;

	return ret;
}

bool CHI2HD_destroyArray(cuMyArray2D *arr){
	if(arr->_array){
		cudaError_t err = cudaFree(arr->_array);
		if(err != cudaSuccess)
			exit(-1);
		return true;
	}
	free(arr->_host_array);

	arr->_array = 0;
	arr->_host_array = 0;
	arr->_device = -1;
	arr->_sizeX = 0;
	arr->_sizeY = 0;
	return false;
}

/******************
 * RESET
 ******************/
__global__ void __CHI2HD_reset(float* arr, int size, float def){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		arr[idx] = def;
}

void CHI2HD_reset(cuMyArray2D *arr, float def){
	dim3 dimGrid(_findOptimalGridSize(arr));
	dim3 dimBlock(_findOptimalBlockSize(arr));
	__CHI2HD_reset<<<dimGrid, dimBlock>>>(arr->_array, arr->getSize(), def);
	cudaThreadSynchronize();
}

/******************
 * SQUARE
 ******************/
__global__ void __CHI2HD_squareIt(float* arr, unsigned int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		arr[idx] = arr[idx] * arr[idx];
}

void CHI2HD_squareIt(cuMyArray2D *arr){
	dim3 dimGrid(_findOptimalGridSize(arr));
	dim3 dimBlock(_findOptimalBlockSize(arr));
	__CHI2HD_squareIt<<<dimGrid, dimBlock>>>(arr->_array, arr->getSize());
	cudaThreadSynchronize();
}

/******************
 * CUBE
 ******************/
__global__ void __CHI2HD_cubeIt(float* arr, int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		arr[idx] = arr[idx] * arr[idx] * arr[idx];
}

void CHI2HD_cubeIt(cuMyArray2D *arr){
	dim3 dimGrid(_findOptimalGridSize(arr));
	dim3 dimBlock(_findOptimalBlockSize(arr));
	__CHI2HD_cubeIt<<<dimGrid, dimBlock>>>(arr->_array, arr->getSize());
	cudaThreadSynchronize();
}

void CHI2HD_copyToHost(cuMyArray2D *arr){
	size_t size = arr->getSize()*sizeof(float);
	if(!arr->_host_array)
		arr->_host_array = (float*)malloc(size);
	cudaError_t err = cudaMemcpy(arr->_host_array, arr->_array, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
		exit(-1);
}

#if defined(__cplusplus)
}
#endif
