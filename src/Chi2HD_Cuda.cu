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
 * Cuda Cores = 240 (30 Multiprosessors)* (8 CUDA Cores/MP)
 * Max Threads x Bloque = 512
 * Maximas dimensiones por bloque = 512 x 512 x 64
 * Maximas dimensiones de una grilla = 65535 x 65535 x 1
 */

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Numeros de bloques a ejecutar.
 */
unsigned int _findOptimalGridSize(cuMyArray2D *arr){
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	unsigned int maxThreads = deviceProp.maxThreadsPerBlock;
	unsigned int requiredGrid = ceil(arr->getSize()/maxThreads)+1;
	if(requiredGrid < deviceProp.maxGridSize[0])
		return requiredGrid;
	return deviceProp.maxGridSize[0];
}

/**
 * Cantidad de threads dentro de cada bloque.
 */
unsigned int _findOptimalBlockSize(cuMyArray2D *arr){
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	return deviceProp.maxThreadsPerBlock;
}

cuMyArray2D CHI2HD_createArray(unsigned int sx, unsigned int sy){
	cuMyArray2D ret;
	cudaError_t err = cudaMalloc((void**)&ret._device_array, (size_t)(sy*sx*sizeof(float)));

	if(err != cudaSuccess)
		exit(1);

	ret._host_array = 0;
	ret._device = 0;
	ret._sizeX = sx;
	ret._sizeY = sy;

	return ret;
}

void CHI2HD_createArrayPointer(unsigned int sx, unsigned int sy, cuMyArray2D* ret){
	cudaError_t err = cudaMalloc((void**)&ret->_device_array, (size_t)(sy*sx*sizeof(float)));

	if(err != cudaSuccess)
		exit(1);

	ret->_host_array = 0;
	ret->_device = 0;
	ret->_sizeX = sx;
	ret->_sizeY = sy;
}

bool CHI2HD_destroyArray(cuMyArray2D *arr){
	if(arr->_device_array){
		cudaError_t err = cudaFree(arr->_device_array);
		if(err != cudaSuccess)
			exit(-1);
		return true;
	}
	if(arr->_host_array)
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
void CHI2HD_copy(cuMyArray2D *src, cuMyArray2D *dst){
	if(!dst->_device_array){
		 cuMyArray2D tmp = CHI2HD_createArray(src->_sizeX, src->_sizeY);
		 dst->_device_array = tmp._device_array;
		 dst->_sizeX = tmp._sizeX;
		 dst->_sizeY = tmp._sizeY;
	}
	cudaError_t err = cudaMemcpy(dst->_device_array, src->_device_array, src->getSize()*sizeof(float), cudaMemcpyDeviceToDevice);
	if(err != cudaSuccess){
		printf("Error: %s", cudaGetErrorString(err));
		exit(-1);
	}
}

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
		err = cudaMalloc((void**)&arr->_device_array, arr->_sizeX*sizeof(float)*arr->_sizeY);
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
	return kernel;
}

/******************
 * Convolucion 2D
 ******************/
__global__ void __CHI2HD_gen_fftresutl(cufftComplex* img, cufftComplex* kernel, float nwnh, unsigned int limit){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < limit){
		float f1 = img[idx].x*kernel[idx].x - img[idx].y*kernel[idx].y;
		float f2 = img[idx].x*kernel[idx].y + img[idx].y*kernel[idx].x;

		img[idx].x=f1/nwnh;
		img[idx].y=f2/nwnh;
	}
}

__global__ void __CHI2HD_copyInside(cufftReal* container, unsigned int container_sizeX, float* data, unsigned int data_sizeX, unsigned int data_sizeY){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int add = floorf(idx/data_sizeX)*(container_sizeX-data_sizeX);
	if(idx < data_sizeX*data_sizeY){
		container[idx+add] = data[idx];
	}
}

void CHI2HD_conv2D(cuMyArray2D* img, cuMyArray2D* kernel_img, cuMyArray2D* output){
	cufftHandle plan_forward_image, plan_forward_kernel, plan_backward;
	cufftComplex *fft_image, *fft_kernel;
	cufftReal *ifft_result, *data, *kernel; // float *

	int nwidth 	=	(int)(img->_sizeX+kernel_img->_sizeX-1);
	int nheight	=	(int)(img->_sizeY+kernel_img->_sizeY-1);
	// Input Complex Data
	cudaMalloc((void**)&fft_image, sizeof(cufftComplex)*(nwidth*(floor(nheight/2) + 1)));
	cudaMalloc((void**)&fft_kernel, sizeof(cufftComplex)*(nwidth*(floor(nheight/2) + 1)));
	// Output Real Data
	cudaMalloc((void**)&ifft_result, sizeof(cufftReal)*nwidth*nheight);
	cudaMalloc((void**)&data, sizeof(cufftReal)*nwidth*nheight);
	cudaMalloc((void**)&kernel, sizeof(cufftReal)*nwidth*nheight);

	// Plans
	cufftPlan2d(&plan_forward_image, nwidth, nheight, CUFFT_R2C);
	cufftPlan2d(&plan_forward_kernel, nwidth, nheight, CUFFT_R2C);
	cufftPlan2d(&plan_backward, nwidth, nheight, CUFFT_C2R);

	// Populate Data
	cudaMemset((void*)data, 0, nwidth*nheight*sizeof(float));
	cudaMemset((void*)kernel, 0, nwidth*nheight*sizeof(float));

	dim3 dimGrid0(_findOptimalGridSize(img));
	dim3 dimBlock0(_findOptimalBlockSize(img));
	__CHI2HD_copyInside<<<dimGrid0, dimBlock0>>>(data, nwidth, img->_device_array, img->_sizeX, img->_sizeY);
	cudaDeviceSynchronize();

	dim3 dimGrid1(_findOptimalGridSize(kernel_img));
	dim3 dimBlock1(_findOptimalBlockSize(kernel_img));
	__CHI2HD_copyInside<<<dimGrid1, dimBlock1>>>(kernel, nwidth, kernel_img->_device_array, kernel_img->_sizeX, kernel_img->_sizeY);
	cudaDeviceSynchronize();

	/* FFT Execute */
		// Execute Plan
		cufftResult res = cufftExecR2C(plan_forward_image, data, fft_image);
		if(res != CUFFT_SUCCESS){
			printf("Error: FFT ");
			exit(-1);
		}

		res = cufftExecR2C(plan_forward_kernel, kernel, fft_kernel);
		if(res != CUFFT_SUCCESS){
			printf("Error: FFT ");
			exit(-1);
		}

		// Populate final data
		dim3 dimGrid2(_findOptimalGridSize(output));
		dim3 dimBlock2(_findOptimalBlockSize(output));
		__CHI2HD_gen_fftresutl<<<dimGrid1, dimBlock1>>>(fft_image, fft_kernel, (float)(nwidth*nheight), (unsigned int)(nwidth * (floor(nheight/2) + 1)));
		cudaDeviceSynchronize();

		// Execute Plan
		res = cufftExecC2R(plan_backward, fft_image, ifft_result);
		if(res != CUFFT_SUCCESS){
			printf("Error: FFT ");
			exit(-1);
		}
	/* FFT Execute */

	// Copy Result to output;
	if(output->_sizeX == (unsigned int)nwidth && output->_sizeY == (unsigned int)nheight){
		cudaMemcpy(output->_device_array, ifft_result, sizeof(cufftReal)*nwidth*nheight, cudaMemcpyDeviceToDevice);
	}

	cufftDestroy(plan_forward_image);
	cufftDestroy(plan_forward_kernel);
	cufftDestroy(plan_backward);
	cudaFree(data);
	cudaFree(kernel);
	cudaFree(ifft_result);
	cudaFree(fft_image);
	cudaFree(fft_kernel);
}

__global__ void __CHI2HD_fftresutl(float* first_term, float* second_term, float* third_term, float* output, unsigned int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		output[idx] = 1.0f/(1.0f + (-2.0f*first_term[idx] + second_term[idx])/third_term[idx]);
	}
}

void CHI2HD_fftresutl(cuMyArray2D* first_term, cuMyArray2D* second_term, cuMyArray2D* third_term, cuMyArray2D* output){
	dim3 dimGrid(_findOptimalGridSize(output));
	dim3 dimBlock(_findOptimalBlockSize(output));
	__CHI2HD_fftresutl<<<dimGrid, dimBlock>>>(first_term->_device_array, second_term->_device_array, third_term->_device_array, output->_device_array, output->getSize());
	cudaDeviceSynchronize();
}

#if defined(__cplusplus)
}
#endif
