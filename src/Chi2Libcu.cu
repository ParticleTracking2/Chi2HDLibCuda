/*
 * Chi2Libcu.cu
 *
 *  Created on: 10/12/2011
 *      Author: juanin
 */

#include "Headers/Chi2Libcu.h"
#include "Headers/Chi2LibcuUtils.h"

/******************
 * Min Max
 ******************/
pair<float, float> Chi2Libcu::minMax(cuMyMatrix *arr){
	pair<float, float> ret;
	arr->copyToHost();

	float tempMax = arr->getValueHost(0);
	float tempMin = arr->getValueHost(0);
	for(unsigned int i=0; i < arr->size(); ++i){
		float tmp = arr->getValueHost(i);
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
__global__ void __Chi2Libcu_normalize(float* arr, unsigned int size, float _min, float _max){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float dif = _max - _min;
	if(idx < size)
		arr[idx] = (float)((1.0f*_max - arr[idx]*1.0f)/dif);
}

void Chi2Libcu::normalize(cuMyMatrix *arr, float _min, float _max){
	dim3 dimGrid(_findOptimalGridSize(arr->size()));
	dim3 dimBlock(_findOptimalBlockSize(arr->size()));
	__Chi2Libcu_normalize<<<dimGrid, dimBlock>>>(arr->devicePointer(), arr->size(), _min, _max);
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);
}


/******************
 * Kernel
 ******************/
__global__ void __Chi2Libcu_gen_kernel(float* arr, unsigned int size, unsigned int ss, unsigned int os, float d, float w){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		float absolute = abs(sqrtf( (idx%ss-os)*(idx%ss-os) + (idx/ss-os)*(idx/ss-os) ));
		arr[idx] = (1.0f - tanhf((absolute - d/2.0f)/w))/2.0f;
	}
}

cuMyMatrix Chi2Libcu::gen_kernel(unsigned int ss, unsigned int os, float d, float w){
	cuMyMatrix kernel(ss,ss);
	dim3 dimGrid(1);
	dim3 dimBlock(ss*ss);
	__Chi2Libcu_gen_kernel<<<dimGrid, dimBlock>>>(kernel.devicePointer(), kernel.size(), ss, os, d, w);
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);

	return kernel;
}

/******************
 * Peaks
 ******************/
__global__ void __Chi2Libcu_findMinimums(float* arr, unsigned int sizeX, unsigned int sizeY, int threshold, int minsep, bool* out){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int imgX = idx%sizeX;
	int imgY = (unsigned int)floorf(idx/sizeY);

	if(idx < sizeX*sizeY && arr[idx] > threshold){
		// Find local Minimum
		for(int localX = minsep; localX >= -minsep; --localX){
			for(int localY = minsep; localY >= -minsep; --localY){
				if(!(localX == 0 && localY == 0)){
					int currentX = (imgX+localX);
					int currentY = (imgY+localY);

					if(currentX < 0)
						currentX = sizeX + currentX;
					if(currentY < 0)
						currentY = sizeY + currentY;

					currentX = (currentX)% sizeX;
					currentY = (currentY)% sizeY;

					if(arr[idx] <= arr[currentX+currentY*sizeY]){
						out[idx] = false;
						return;
					}
				}
			}
		}
		out[idx] = true;
		return;
	}
	out[idx] = false;
}

__global__ void __Chi2Libcu_fillPeakArray(float* img, bool* peaks_detected, unsigned int sizeX, unsigned int sizeY, cuMyPeak* peaks){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Lineal
	if(idx == 0){
		unsigned int pindex = 0;
		for(unsigned int i = 0; i < sizeX*sizeY; ++i){
			if(peaks_detected[i]){
				int imgX = i%sizeX;
				int imgY = (unsigned int)floorf(idx/sizeY);

				cuMyPeak tmp;
				tmp.valid = true;
				tmp.lineal_index = i;
				tmp.x = imgX;	tmp.y = imgY;
				tmp.chi_intensity = img[i];
				peaks[pindex] = tmp;
			}
		}
	}
}

__global__ void __Chi2Libcu_validatePeaks(cuMyPeak* peaks, unsigned int size, unsigned int mindistance){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int mindistance2 = mindistance*mindistance;

	if(idx < size)
	for(unsigned int j=idx+1; j < size; ++j){
		int difx = peaks[idx].x - peaks[j].x;
		int dify = peaks[idx].y - peaks[j].y;

		if( (difx*difx + dify*dify) < mindistance2){
			peaks[idx].valid = false;
			break;
		}
	}
}

void Chi2Libcu::getPeaks(cuMyMatrix *arr, int threshold, int mindistance, int minsep){
	bool* d_minimums;
	bool* h_minimums;
	size_t arrSize = arr->size()*sizeof(bool);
	cudaError_t err = cudaMalloc((void**)&d_minimums, arrSize);
	manageError(err);

	// Encontrar Minimos
	dim3 dimGrid(_findOptimalGridSize(arr->size()));
	dim3 dimBlock(_findOptimalBlockSize(arr->size()));
	__Chi2Libcu_findMinimums<<<dimGrid, dimBlock>>>(arr->devicePointer(), arr->sizeX(), arr->sizeY(), threshold, minsep, d_minimums);
	err = cudaDeviceSynchronize();
	manageError(err);

	// Contar minimos
	h_minimums = (bool*)malloc(arrSize);
	err = cudaMemcpy(h_minimums, d_minimums, arrSize, cudaMemcpyDeviceToHost);
	manageError(err);
	unsigned int counter = 0;
	for(unsigned int i=0; i < arr->size(); ++i){
		if(h_minimums[i])
			counter++;
	}

	// Alocar datos
	// TODO: Idea: Almacenar Peaks encontrados en Memoria compartida y cuando acaben copiar a memoria Global.
	cuMyPeakArray peaks;
	peaks.size = counter;
	err = cudaMalloc((void**)&peaks._device_array, counter*sizeof(cuMyPeak));
	manageError(err);

	dim3 dimGrid1(1); dim3 dimBlock1(1);
	__Chi2Libcu_fillPeakArray<<<dimGrid1, dimBlock1>>>(arr->devicePointer(), d_minimums, arr->sizeX(), arr->sizeY(), peaks._device_array);
	err = cudaDeviceSynchronize();
	manageError(err);

	// Ordenar de menor a mayor en intensidad de imagen Chi
	// TODO: Hacer un algoritmo de ordenamiento

	// Validar
	dim3 dimGrid2(_findOptimalGridSize(peaks.size));
	dim3 dimBlock2(_findOptimalBlockSize(peaks.size));
	__Chi2Libcu_validatePeaks<<<dimGrid2, dimBlock2>>>(peaks._device_array, peaks.size, mindistance);
	err = cudaDeviceSynchronize();
	manageError(err);

	// Compactar, eliminar los peaks no validos
	// TODO

	printf("Total Minimums : %i of %i\n", counter, arr->size());
	cudaFree(d_minimums);
	free(h_minimums);
}
