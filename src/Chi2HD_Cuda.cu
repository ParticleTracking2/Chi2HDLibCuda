//============================================================================
// Name        : Chi2HD_Cuda.cu
// Author      : Juan Silva
// Version     :
// Copyright   : All rights reserved
// Description : Funciones para trabajar el algoritmo de minimos cuadrados
//============================================================================

#include "Headers/Chi2HD_Cuda.h"
#include "cufft.h"

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
	cudaError_t err = cudaMallocPitch((void**)&ret._device_array, &ret._device_pitch, (size_t)(sx*sizeof(float)), (size_t)(sy));

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
		err = cudaMallocPitch((void**)&arr->_device_array, &arr->_device_pitch, arr->_sizeX*sizeof(float), arr->_sizeY);
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

/******************
 * Convolucion 2D
 ******************/
cuMyArray2D CHI2HD_conv2D(cuMyArray2D* img, cuMyArray2D* kernel_img){
	cufftHandle plan_forward_image, plan_forward_kernel, plan_backward;
	cufftComplex *fft_image, *fft_kernel;
	cufftReal *ifft_result, *data, *kernel; // float *
	size_t ifft_result_pitch, data_pitch, kernel_pitch;

	int nwidth 	=	(int)(img->_sizeX+kernel_img->_sizeX-1);
	int nheight	=	(int)(img->_sizeY+kernel_img->_sizeY-1);
	// Input Complex Data
	cudaMalloc((void**)&fft_image, sizeof(cufftComplex)*(nwidth*(floor(nheight/2) + 1)));
	cudaMalloc((void**)&fft_kernel, sizeof(cufftComplex)*(nwidth*(floor(nheight/2) + 1)));
	// Output Real Data
	cudaMallocPitch((void**)&ifft_result, &ifft_result_pitch, sizeof(cufftReal)*nwidth, nheight);
	cudaMallocPitch((void**)&data, &data_pitch, sizeof(cufftReal)*nwidth, nheight);
	cudaMallocPitch((void**)&kernel, &kernel_pitch, sizeof(cufftReal)*nwidth, nheight);

	// Plans
	cufftPlan2d(&plan_forward_image, nwidth, nheight, CUFFT_R2C);
	cufftPlan2d(&plan_forward_kernel, nwidth, nheight, CUFFT_R2C);
	cufftPlan2d(&plan_backward, nwidth, nheight, CUFFT_C2R);

	// Populate Data
	cudaMemset((void*)data, 0, nwidth*nheight*sizeof(float));
	cudaMemset((void*)kernel, 0, nwidth*nheight*sizeof(float));
	cudaMemcpy2D(data, data_pitch, img->_device_array, img->_device_pitch, img->_sizeX*sizeof(float), img->_sizeY, cudaMemcpyDeviceToDevice);
	cudaMemcpy2D(kernel, kernel_pitch, kernel_img->_device_array, kernel_img->_device_pitch, kernel_img->_sizeX*sizeof(float), kernel_img->_sizeY, cudaMemcpyDeviceToDevice);

	// Execute Plan
	cufftExecR2C(plan_forward_image, data, fft_image);

	cufftExecR2C(plan_forward_kernel, kernel, fft_kernel);

	// Populate final data
	// TODO

	// Execute Plan
	cufftExecC2R(plan_backward, fft_image, ifft_result);

	// Copy Result to output;
	// TODO
	cuMyArray2D ret;

	cufftDestroy(plan_forward_image);
	cufftDestroy(plan_forward_kernel);
	cufftDestroy(plan_backward);
	cudaFree(data);
	cudaFree(kernel);
	cudaFree(ifft_result);
	cudaFree(fft_image);
	cudaFree(fft_kernel);

	return ret;
}

#if defined(__cplusplus)
}
#endif
