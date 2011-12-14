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

/******************
 * Check Inside Peaks & Append
 ******************/
__global__ void __checkInside(cuMyPeak* arr, unsigned int size, unsigned int sizeX, unsigned int sizeY, int pad){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= size)
		return;

	if(	0 <= (arr[idx].x - pad) && (arr[idx].x - pad) < (int)sizeX &&
		0 <= (arr[idx].y - pad) && (arr[idx].y - pad) < (int)sizeY){
		arr[idx].valid = true;
	}else{
		arr[idx].valid = false;
	}
}
unsigned int Chi2LibcuHighDensity::checkInsidePeaks(cuMyPeakArray *old_peaks, cuMyPeakArray *new_peaks, cuMyMatrix *img, unsigned int os){

	// Validar al interior los nuevos Peaks;
	dim3 dimGrid(_findOptimalGridSize(new_peaks->size()));
	dim3 dimBlock(_findOptimalBlockSize(new_peaks->size()));
	__checkInside<<<dimGrid, dimBlock>>>(new_peaks->devicePointer(), new_peaks->size(), img->sizeX(), img->sizeY(), (int)os);
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);

	new_peaks->keepValids();
	unsigned int total_new = new_peaks->size();
	old_peaks->append(new_peaks);

	return total_new;
}
