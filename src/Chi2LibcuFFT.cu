/*
 * Chi2LibcuFFT.cu
 *
 *  Created on: 10/12/2011
 *      Author: juanin
 */

#include "Headers/Chi2LibcuFFT.h"
#include "Headers/Chi2LibcuUtils.h"

/**
 * Maneja los errores de CUDA
 */
void Chi2LibcuFFT::manageErrorFFT(cufftResult res){
	if(res != CUFFT_SUCCESS){
		printf("CHI2HD_CUDA FFT Error: ");
		switch (res) {
			case CUFFT_INVALID_PLAN:
				printf("Plan Invalido\n");
				break;
			case CUFFT_INVALID_TYPE:
				printf("Tipo invalido\n");
				break;
			case CUFFT_INVALID_VALUE:
				printf("Valor invalido\n");
				break;
			case CUFFT_INTERNAL_ERROR:
				printf("Error inerno\n");
				break;
			case CUFFT_EXEC_FAILED:
				printf("Falla de ejecucion\n");
				break;
			case CUFFT_SETUP_FAILED:
				printf("Falla de setup\n");
				break;
			case CUFFT_INVALID_SIZE:
				printf("Tamaño invalido\n");
				break;
			case CUFFT_UNALIGNED_DATA:
				printf("Datos desalineados\n");
				break;
			default:
				printf("Error Desconocido\n");
				break;
		}
		exit(-1);
	}
}

/******************
 * Modula y Normaliza cada elemento de la transformacion.
 * Guarda los resultados en img.
 ******************/
__global__ void __Chi2LibcuFFT_modulateAndNormalize(cufftComplex* img, cufftComplex* kernel, float nwnh, int limit){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < limit){
		float f1 = img[idx].x*kernel[idx].x - img[idx].y*kernel[idx].y;
		float f2 = img[idx].x*kernel[idx].y + img[idx].y*kernel[idx].x;

		img[idx].x=f1*nwnh;
		img[idx].y=f2*nwnh;
	}
}

/******************
 * Copia la matriz transpuesta
 ******************/
__global__ void __Chi2LibcuFFT_copyInside(cufftReal* container, unsigned int container_sizeX, unsigned int container_sizeY, float* data, unsigned int data_sizeX, unsigned int data_sizeY){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	int add = floorf(idx/data_sizeX)*(container_sizeX-data_sizeX); // Normal
	int add = floorf(idx%data_sizeX)*container_sizeX-idx + floorf(idx/data_sizeX); // Transpuesta
	if(idx < data_sizeX*data_sizeY){
		container[idx+add] = data[idx];
	}
}

/******************
 * Convolucion 2D
 * Usando Zero Padding
 ******************/
void Chi2LibcuFFT::conv2D(cuMyMatrix* img, cuMyMatrix* kernel_img, cuMyMatrix* output){
	cufftHandle plan_forward_image, plan_forward_kernel, plan_backward;
	cufftComplex *fft_image, *fft_kernel;
	cufftReal *ifft_result, *data, *kernel; // float *

	int nwidth 	=	output->sizeX(); //(int)(img->_sizeX+kernel_img->_sizeX-1);
	int nheight	=	output->sizeY(); //(int)(img->_sizeY+kernel_img->_sizeY-1);
	// Input Complex Data
	cudaError_t err;
	err = cudaMalloc((void**)&fft_image, sizeof(cufftComplex)*(nwidth*(nheight/2 +1)));
	manageError(err);
	err = cudaMalloc((void**)&fft_kernel, sizeof(cufftComplex)*(nwidth*(nheight/2 +1)));
	manageError(err);
	// Output Real Data
	err = cudaMalloc((void**)&ifft_result, sizeof(cufftReal)*nwidth*nheight);
	manageError(err);
	err = cudaMalloc((void**)&data, sizeof(cufftReal)*nwidth*nheight);
	manageError(err);
	err = cudaMalloc((void**)&kernel, sizeof(cufftReal)*nwidth*nheight);
	manageError(err);

	// Plans
	cufftResult res = cufftPlan2d(&plan_forward_image, nwidth, nheight, CUFFT_R2C);
	manageErrorFFT(res);
	res = cufftPlan2d(&plan_forward_kernel, nwidth, nheight, CUFFT_R2C);
	manageErrorFFT(res);
	res = cufftPlan2d(&plan_backward, nwidth, nheight, CUFFT_C2R);
	manageErrorFFT(res);

	// Populate Data
	err = cudaMemset((void*)data, 0, nwidth*nheight*sizeof(cufftReal));
	manageError(err);
	err = cudaMemset((void*)kernel, 0, nwidth*nheight*sizeof(cufftReal));
	manageError(err);

	dim3 dimGrid0(_findOptimalGridSize(img->size()));
	dim3 dimBlock0(_findOptimalBlockSize(img->size()));
	__Chi2LibcuFFT_copyInside<<<dimGrid0, dimBlock0>>>(data, nwidth, nheight, img->devicePointer(), img->sizeX(), img->sizeY());
	checkAndSync();

	dim3 dimGrid1(_findOptimalGridSize(kernel_img->size()));
	dim3 dimBlock1(_findOptimalBlockSize(kernel_img->size()));
	__Chi2LibcuFFT_copyInside<<<dimGrid1, dimBlock1>>>(kernel, nwidth, nheight, kernel_img->devicePointer(), kernel_img->sizeX(), kernel_img->sizeY());
	checkAndSync();

	/* FFT Execute */
		// Execute Plan
		res = cufftExecR2C(plan_forward_image, data, fft_image);
		manageErrorFFT(res);
		err = cudaDeviceSynchronize();
		manageError(err);

		res = cufftExecR2C(plan_forward_kernel, kernel, fft_kernel);
		manageErrorFFT(res);
		err = cudaDeviceSynchronize();
		manageError(err);

		// Modular y Normalizar
		dim3 dimGrid2(_findOptimalGridSize(output->size()));
		dim3 dimBlock2(_findOptimalBlockSize(output->size()));
		__Chi2LibcuFFT_modulateAndNormalize<<<dimGrid2, dimBlock2>>>(fft_image, fft_kernel, (float)(1.0f/(float)(nwidth*nheight)), (int)(nwidth *(nheight/2 +1)));
		checkAndSync();

		// Execute Plan
		res = cufftExecC2R(plan_backward, fft_image, ifft_result);
		manageErrorFFT(res);
		err = cudaDeviceSynchronize();
		manageError(err);
	/* FFT Execute */

	// Copy Result to output;
	err = cudaMemcpy(output->devicePointer(), ifft_result, sizeof(cufftReal)*nwidth*nheight, cudaMemcpyDeviceToDevice);
	manageError(err);

	cufftDestroy(plan_forward_image);
	cufftDestroy(plan_forward_kernel);
	cufftDestroy(plan_backward);
	err = cudaFree(data); manageError(err);
	err = cudaFree(kernel); manageError(err);
	err = cudaFree(ifft_result); manageError(err);
	err = cudaFree(fft_image); manageError(err);
	err = cudaFree(fft_kernel); manageError(err);
}

/******************
 * Calcula el resultado general de las transformaciones
 ******************/
__global__ void __Chi2LibcuFFT_fftresutl(float* first_term, float* second_term, float* third_term, float* output, unsigned int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		output[idx] = 1.0f/(1.0f + (-2.0f*first_term[idx] + second_term[idx])/third_term[idx]);
	}
}

void Chi2LibcuFFT::fftresutl(cuMyMatrix* first_term, cuMyMatrix* second_term, cuMyMatrix* third_term, cuMyMatrix* output){
	dim3 dimGrid(_findOptimalGridSize(output->size()));
	dim3 dimBlock(_findOptimalBlockSize(output->size()));
	__Chi2LibcuFFT_fftresutl<<<dimGrid, dimBlock>>>(first_term->devicePointer(), second_term->devicePointer(), third_term->devicePointer(), output->devicePointer(), output->size());
	checkAndSync();
}


