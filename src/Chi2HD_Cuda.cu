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
 * CUBE
 ******************/
__global__ void __CHI2HD_minMax(float* arr, int size, float* maxData, float* minData){
	__shared__ float _min[256];
	__shared__ float _max[256];

	int idx = 128*256*blockIdx.y + 256*blockIdx.x + threadIdx.x;
	_min[threadIdx.x] = _max[threadIdx.x] = arr[idx];
	__syncthreads();
	int nTotalThreads = blockDim.x;
	while(nTotalThreads > 1){
		int halfPoint = (nTotalThreads >> 1);
		if (threadIdx.x < halfPoint){
			float temp;
			// Minimo
			temp = _min[threadIdx.x + halfPoint];
			if (temp < _min[threadIdx.x]) _min[threadIdx.x] = temp;

			// Maximo
			temp = _max[threadIdx.x + halfPoint];
			if (temp > _max[threadIdx.x]) _max[threadIdx.x] = temp;
		}
		__syncthreads();
		nTotalThreads = (nTotalThreads >> 1);
	}

	if (threadIdx.x == 0){
		maxData[128*blockIdx.y + blockIdx.x] = _max[0];
		minData[128*blockIdx.y + blockIdx.x] = _min[0];
	}
}

myPair CHI2HD_minMax(cuMyArray2D *arr){
	int MAX_DATA_SIZE = arr->getSize();
	int THREADS_PER_BLOCK = 256;
	int BLOCKS_PER_GRID_ROW = 128;
	myPair ret; ret._min = 0; ret._max = 0;
	// Host
	float * h_resultMax = (float *)malloc(sizeof(float) * MAX_DATA_SIZE/THREADS_PER_BLOCK);
	float * h_resultMin = (float *)malloc(sizeof(float) * MAX_DATA_SIZE/THREADS_PER_BLOCK);

	// Device
	float * d_resultMax;
	cudaError_t err = cudaMalloc( (void **)&d_resultMax, sizeof(float) * MAX_DATA_SIZE/THREADS_PER_BLOCK);
	if(err != cudaSuccess) exit(-1);
	float * d_resultMin;
	err = cudaMalloc( (void **)&d_resultMin, sizeof(float) * MAX_DATA_SIZE/THREADS_PER_BLOCK);
	if(err != cudaSuccess) exit(-1);

	int blockGridWidth = BLOCKS_PER_GRID_ROW;
	int blockGridHeight = (MAX_DATA_SIZE / THREADS_PER_BLOCK) / blockGridWidth;

	dim3 blockGridRows(blockGridWidth, blockGridHeight);
	dim3 threadBlockRows(THREADS_PER_BLOCK, 1);
	__CHI2HD_minMax<<<blockGridRows, threadBlockRows>>>(arr->_device_array, arr->getSize(), d_resultMax, d_resultMin);
	cudaDeviceSynchronize();

	cudaMemcpy(h_resultMin, d_resultMin, sizeof(float) * MAX_DATA_SIZE/THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_resultMax, d_resultMax, sizeof(float) * MAX_DATA_SIZE/THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);

	float tempMin = h_resultMin[0];
	float tempMax = h_resultMax[0];
	int i;
	for (i=1 ; i < MAX_DATA_SIZE/THREADS_PER_BLOCK; i++){
		if (h_resultMin[i] < tempMin) tempMin = h_resultMin[i];
		if (h_resultMax[i] > tempMax) tempMax = h_resultMax[i];
	}

	free(h_resultMax);free(h_resultMin);
	err = cudaFree(d_resultMin);
	if(err != cudaSuccess) exit(-1);
	err = cudaFree(d_resultMax);
	if(err != cudaSuccess) exit(-1);

	ret._max = tempMax;
	ret._min = tempMin;
	return ret;
}

#if defined(__cplusplus)
}
#endif
