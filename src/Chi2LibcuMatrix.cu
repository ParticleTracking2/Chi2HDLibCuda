/*
 * Chi2LibcuMatrix.cu
 *
 *  Created on: 09/12/2011
 *      Author: juanin
 */

#include "Headers/Chi2LibcuMatrix.h"
#include "Headers/Chi2LibcuUtils.h"

/******************
 * SQUARE
 ******************/
__global__ void __Chi2LibcuMatrix_squareIt(float* arr, unsigned int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		arr[idx] = arr[idx] * arr[idx];
}

void Chi2LibcuMatrix::squareIt(cuMyMatrix *mtrx){
	dim3 dimGrid(_findOptimalGridSize(mtrx->size()));
	dim3 dimBlock(_findOptimalBlockSize(mtrx->size()));
	__Chi2LibcuMatrix_squareIt<<<dimGrid, dimBlock>>>(mtrx->devicePointer(), mtrx->size());
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);
}

/******************
 * CUBE
 ******************/
__global__ void __Chi2LibcuMatrix_cubeIt(float* arr, unsigned int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		arr[idx] = arr[idx] * arr[idx] * arr[idx];
}

void Chi2LibcuMatrix::cubeIt(cuMyMatrix *mtrx){
	dim3 dimGrid(_findOptimalGridSize(mtrx->size()));
	dim3 dimBlock(_findOptimalBlockSize(mtrx->size()));
	__Chi2LibcuMatrix_cubeIt<<<dimGrid, dimBlock>>>(mtrx->devicePointer(), mtrx->size());
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);
}

/******************
 * Copy
 ******************/
void Chi2LibcuMatrix::copy(cuMyMatrix *in, cuMyMatrix *out){
	if(!in->devicePointer()){
		 out->allocateDevice(in->sizeX(), in->sizeY());
	}
	cudaError_t err = cudaMemcpy(out->devicePointer(), in->devicePointer(), in->size()*sizeof(float), cudaMemcpyDeviceToDevice);
	manageError(err);
}

void Chi2LibcuMatrix::copy(cuMyMatrixi *in, cuMyMatrixi *out){
	if(!in->devicePointer()){
		 out->allocateDevice(in->sizeX(), in->sizeY());
	}
	cudaError_t err = cudaMemcpy(out->devicePointer(), in->devicePointer(), in->size()*sizeof(int), cudaMemcpyDeviceToDevice);
	manageError(err);
}
