//============================================================================
// Name        : Chi2HD_Cuda.cu
// Author      : Juan Silva
// Version     :
// Copyright   : All rights reserved
// Description : Funciones para trabajar el algoritmo de minimos cuadrados
//============================================================================

#include "Headers/Chi2HD_Cuda.h"

/**
 * Elementos a considerar con 275 GTX
 * Global Memory = 896 MB
 * Const Memory = 65KB (16000 Floats)
 * Shared Memory = 16KB (Compartida en el bloque = 4000 Floats)
 * Registros por Bloque = 16K
 * ------------------------------
 * Cuda Cores = 240
 * Max Threads x Bloque = 512
 * Maximas dimensiones por bloque = 512 x 512 x 64
 * Maximas dimensiones de una grilla = 65535 x 65535 x 1
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
	cudaError_t err = cudaMalloc((void**)&ret._device_array, (size_t) sx*sy*sizeof(float));

	if(err != cudaSuccess)
		exit(1);

	ret._host_array = 0;
	ret._device = 0;
	ret._sizeX = sx;
	ret._sizeY = sy;

	return ret;
}

bool CHI2HD_destroyArray(cuMyArray2D *arr){
	if(arr->_device_array){
		cudaError_t err = cudaFree(arr->_device_array);
		if(err != cudaSuccess)
			exit(-1);
		return true;
	}
	free(arr->_host_array);

	arr->_device_array = 0;
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
	__CHI2HD_reset<<<dimGrid, dimBlock>>>(arr->_device_array, arr->getSize(), def);
	cudaDeviceSynchronize();
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
	__CHI2HD_squareIt<<<dimGrid, dimBlock>>>(arr->_device_array, arr->getSize());
	cudaDeviceSynchronize();
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
	__CHI2HD_cubeIt<<<dimGrid, dimBlock>>>(arr->_device_array, arr->getSize());
	cudaDeviceSynchronize();
}

/******************
 * Copy
 ******************/
void CHI2HD_copyToHost(cuMyArray2D *arr){
	size_t size = arr->getSize()*sizeof(float);
	if(!arr->_host_array)
		arr->_host_array = (float*)malloc(size);
	cudaError_t err = cudaMemcpy(arr->_host_array, arr->_device_array, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
		exit(-1);
}

void CHI2HD_copyToDevice(cuMyArray2D *arr){
	size_t size = arr->getSize()*sizeof(float);
	cudaError_t err;
	if(!arr->_device_array){
		err = cudaMalloc((void**)&arr->_device_array, size);
		if(err != cudaSuccess) exit(-1);
	}
	err = cudaMemcpy(arr->_device_array, arr->_host_array, size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) exit(-1);
}

/******************
 * Min Max
 ******************/
myPair CHI2HD_minMax(cuMyArray2D *arr){
	myPair ret;
	if(!arr->_host_array)
		CHI2HD_copyToHost(arr);

	float tempMax = arr->getValueHost(0);
	float tempMin = arr->getValueHost(0);
	for(unsigned int i=0; i < arr->getSize(); ++i){
		float tmp = arr->_host_array[i];
		if(tempMax < tmp)
			tempMax = tmp;
		if(tempMin > tmp)
			tempMin = tmp;
	}
	ret.first = tempMin;
	ret.second = tempMax;

	return ret;
}

/******************
 * Normalizar
 ******************/
__global__ void __CHI2HD_normalize(float* arr, unsigned int size, float _min, float _max){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float dif = _max - _min;
	if(idx < size)
		arr[idx] = (float)((1.0f*_max - arr[idx]*1.0f)/dif);
}

void CHI2HD_normalize(cuMyArray2D *arr, float _min, float _max){
	dim3 dimGrid(_findOptimalGridSize(arr));
	dim3 dimBlock(_findOptimalBlockSize(arr));
	__CHI2HD_normalize<<<dimGrid, dimBlock>>>(arr->_device_array, arr->getSize(), _min, _max);
	cudaDeviceSynchronize();
}

/******************
 * Kernel
 ******************/
__global__ void __CHI2HD_gen_kernel(float* arr, unsigned int size, unsigned int ss, unsigned int os, float d, float w){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		float absolute = abs(sqrtf( (idx%ss-os)*(idx%ss-os) + (idx/ss-os)*(idx/ss-os) ));
		arr[idx] = (1.0f - tanhf((absolute - d/2.0f)/w))/2.0f;
	}
}

cuMyArray2D CHI2HD_gen_kernel(unsigned int ss, unsigned int os, float d, float w){
	cuMyArray2D kernel = CHI2HD_createArray(ss,ss);
	dim3 dimGrid(1);
	dim3 dimBlock(ss*ss);
	__CHI2HD_gen_kernel<<<dimGrid, dimBlock>>>(kernel._device_array, kernel.getSize(), ss, os, d, w);
	cudaDeviceSynchronize();
	CHI2HD_copyToHost(&kernel);
	return kernel;
}

#if defined(__cplusplus)
}
#endif
