//============================================================================
// Name        : Chi2HD_Cuda.cu
// Author      : Juan Silva
// Version     :
// Copyright   : All rights reserved
// Description : Funciones para trabajar el algoritmo de minimos cuadrados
//============================================================================

#include "Headers/Chi2HD_Cuda.h"
#include "Headers/Chi2HD_CudaUtils.h"

/**
 * Elementos a considerar con 275 GTX
 * Global Memory = 896 MB
 * Const Memory = 65KB (16000 Floats)
 * Shared Memory = 16KB (Compartida en el bloque = 4000 Floats)
 * Registros por Bloque = 16K
 * ------------------------------
 * Cuda Cores = 240 (30 Multiprosessors)* (8 CUDA Cores/MP)
 * Max Threads x Bloque = 512
 * Maximas dimensiones por bloque = 512 x 512 x 64
 * Maximas dimensiones de una grilla = 65535 x 65535 x 1
 */

#if defined(__cplusplus)
extern "C" {
#endif

/******************
 * Creacion y destruccion de arreglos
 ******************/
cuMyArray2D CHI2HD_createArray(unsigned int sx, unsigned int sy){
	cuMyArray2D ret;
	cudaError_t err = cudaMalloc((void**)&ret._device_array, (size_t)(sy*sx*sizeof(float)));
	manageError(err);

	ret._host_array = 0;
	ret._device = 0;
	ret._sizeX = sx;
	ret._sizeY = sy;

	return ret;
}

void CHI2HD_createArrayPointer(unsigned int sx, unsigned int sy, cuMyArray2D* ret){
	cudaError_t err = cudaMalloc((void**)&ret->_device_array, (size_t)(sy*sx*sizeof(float)));
	manageError(err);

	ret->_host_array = 0;
	ret->_device = 0;
	ret->_sizeX = sx;
	ret->_sizeY = sy;
}

bool CHI2HD_destroyArray(cuMyArray2D *arr){
	if(arr->_device_array){
		cudaError_t err = cudaFree(arr->_device_array);
		manageError(err);
	}
	if(arr->_host_array)
		free(arr->_host_array);

	arr->_device_array = 0;
	arr->_host_array = 0;
	arr->_device = -1;
	arr->_sizeX = 0;
	arr->_sizeY = 0;
	return true;
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
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);
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
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);
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
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);
}

/******************
 * Copy
 ******************/
void CHI2HD_copy(cuMyArray2D *src, cuMyArray2D *dst){
	if(!dst->_device_array){
		 cuMyArray2D tmp = CHI2HD_createArray(src->_sizeX, src->_sizeY);
		 dst->_device_array = tmp._device_array;
		 dst->_sizeX = tmp._sizeX;
		 dst->_sizeY = tmp._sizeY;
	}
	cudaError_t err = cudaMemcpy(dst->_device_array, src->_device_array, src->getSize()*sizeof(float), cudaMemcpyDeviceToDevice);
	manageError(err);
}

void CHI2HD_copyToHost(cuMyArray2D *arr){
	size_t size = arr->getSize()*sizeof(float);
	if(!arr->_host_array)
		arr->_host_array = (float*)malloc(size);
	cudaError_t err = cudaMemcpy(arr->_host_array, arr->_device_array, size, cudaMemcpyDeviceToHost);
	manageError(err);
}

void CHI2HD_copyToDevice(cuMyArray2D *arr){
	size_t size = arr->getSize()*sizeof(float);
	cudaError_t err;
	if(!arr->_device_array){
		err = cudaMalloc((void**)&arr->_device_array, arr->_sizeX*sizeof(float)*arr->_sizeY);
		manageError(err);
	}
	err = cudaMemcpy(arr->_device_array, arr->_host_array, size, cudaMemcpyHostToDevice);
	manageError(err);
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
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);
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
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);
	return kernel;
}

/******************
 * Peaks
 ******************/
__global__ void __CHI2HD_getPeaks(float* arr, unsigned int sizeX, unsigned int sizeY, int threshold, int minsep, bool* out){
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

void CHI2HD_getPeaks(cuMyArray2D *arr, int threshold, int mindistance, int minsep){
	bool* d_minimums;
	bool* h_minimums;
	size_t arrSize = arr->_sizeX*arr->_sizeY*sizeof(bool);
	cudaError_t err = cudaMalloc((void**)&d_minimums, arrSize);
	manageError(err);

	// Encontrar Minimos
	dim3 dimGrid(_findOptimalGridSize(arr));
	dim3 dimBlock(_findOptimalBlockSize(arr));
	__CHI2HD_getPeaks<<<dimGrid, dimBlock>>>(arr->_device_array, arr->_sizeX, arr->_sizeY, threshold, minsep, d_minimums);
	err = cudaDeviceSynchronize();
	manageError(err);

	// Contar minimos
	h_minimums = (bool*)malloc(arrSize);
	err = cudaMemcpy(h_minimums, d_minimums, arrSize, cudaMemcpyDeviceToHost);
	manageError(err);
	unsigned int counter = 0;
	for(unsigned int i=0; i < arr->getSize(); ++i){
		if(h_minimums[i])
			counter++;
	}

	// Alocar datos

	printf("Total Minimums : %i of %i\n", counter, arr->getSize());
	cudaFree(d_minimums);
	free(h_minimums);
}

#if defined(__cplusplus)
}
#endif
