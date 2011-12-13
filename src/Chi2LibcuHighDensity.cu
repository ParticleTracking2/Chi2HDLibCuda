/*
 * Chi2LibcuHighDensity.cu
 *
 *  Created on: 13/12/2011
 *      Author: juanin
 */

#include "Headers/Chi2LibcuHighDensity.h"
#include "Headers/Chi2LibcuUtils.h"

/******************
 * Scale Image
 ******************/
__global__ void __scaleImage(float* arr, float* out, unsigned int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		out[idx] = 4*arr[idx]*arr[idx];
}

void Chi2LibcuHighDensity::scaleImage(cuMyMatrix* img, cuMyMatrix* out){
	dim3 dimGrid(_findOptimalGridSize(img->size()));
	dim3 dimBlock(_findOptimalBlockSize(img->size()));
	__scaleImage<<<dimGrid, dimBlock>>>(img->devicePointer(), out->devicePointer(), img->size());
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);
}

/******************
 * Invert Image
 ******************/
__global__ void __invertImage(float* arr, unsigned int size, float maxval){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= size)
		return;

	arr[idx] = maxval - arr[idx];
}

void Chi2LibcuHighDensity::invertImage(cuMyMatrix* img, float maxval){
	dim3 dimGrid(_findOptimalGridSize(img->size()));
	dim3 dimBlock(_findOptimalBlockSize(img->size()));
	__invertImage<<<dimGrid, dimBlock>>>(img->devicePointer(), img->size(), maxval);
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);
}
