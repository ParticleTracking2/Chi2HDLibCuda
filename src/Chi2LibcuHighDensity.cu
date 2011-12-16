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
	filterPeaksOutside(new_peaks, img, os);
	new_peaks->keepValids();
	unsigned int total_new = new_peaks->size();
	old_peaks->append(new_peaks);

	return total_new;
}

void Chi2LibcuHighDensity::filterPeaksOutside(cuMyPeakArray *peaks, cuMyMatrix *img, unsigned int os){
	dim3 dimGrid(_findOptimalGridSize(peaks->size()));
	dim3 dimBlock(_findOptimalBlockSize(peaks->size()));
	__checkInside<<<dimGrid, dimBlock>>>(peaks->devicePointer(), peaks->size(), img->sizeX(), img->sizeY(), (int)os);
	cudaError_t err = cudaGetLastError(); manageError(err);
	err = cudaDeviceSynchronize(); manageError(err);

	peaks->keepValids();
}

/******************
 * Gaussian Fit
 ******************/
__global__ void __customCopy(cuMyPeak* src, cuMyPeak* dst, unsigned int peaks_size, float* img, unsigned int sizeX, unsigned int sizeY, unsigned int ss){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= peaks_size)
		return;

	unsigned int _x, _y;
	_x = src[idx].x - ss;
	_y = src[idx].y - ss;

	dst[idx].x = _x;
	dst[idx].y = _y;
	dst[idx].chi_intensity = img[_x+_y*sizeY];
}

std::pair<double, double> Chi2LibcuHighDensity::gaussianFit(cuMyPeakArray *peaks, cuMyMatrix *img, unsigned int ss){
	unsigned int slots = 21;
	double X[slots];
	int freq[slots];
	double dx = 0.01;
	for(unsigned int i=0; i < slots; ++i){
		freq[i]=0;
		X[i]=0.8+i*dx;
	}

	cuMyPeakArray new_peaks(peaks->size());
	dim3 dimGrid(_findOptimalGridSize(peaks->size()));
	dim3 dimBlock(_findOptimalBlockSize(peaks->size()));
	__customCopy<<<dimGrid, dimBlock>>>(peaks->devicePointer(), new_peaks.devicePointer(), peaks->size(), img->devicePointer(), img->sizeX(), img->sizeY(), ss);
	checkAndSync();

	new_peaks.sortByChiIntensity();

	new_peaks.copyToHost();
	unsigned int current;
	for(unsigned int i=0; i < new_peaks.size(); ++i){
		for(unsigned int s=current; s < slots; ++s){
			if(new_peaks.getHostValue(i).chi_intensity <= X[s] + dx){
				++freq[s];
				break;
			}
			++current;
		}
	}

	int check = 0;
	double mu = 0.0, sigma = 0.0;
	for(unsigned int i=0; i < slots; ++i){
		mu += X[i]*freq[i];
		check += freq[i];
	}
	mu = mu/check;

	for(unsigned int i=0; i < slots; ++i){
		sigma += freq[i]*(mu-X[i])*(mu-X[i]);
	}
	sigma = sqrt(sigma/check);

	pair<double, double> ret;
	ret.first = mu; ret.second = sigma;
	return ret;
}
