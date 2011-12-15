/*
 * cuDualData.cu
 *
 *  Created on: 12/12/2011
 *      Author: juanin
 */

#include "../Headers/Container/cuDualData.h"
#include "../Headers/Chi2LibcuUtils.h"

DualDatai DualData_CreateInt(unsigned int size){
	DualDatai pair;

	pair.h_data = (int*)malloc(size*sizeof(int));
	cudaError_t err = cudaMalloc((void**)&pair.d_data, size*sizeof(int));
	manageError(err);

	for(unsigned int i=0; i < size; ++i)
		pair.h_data[i] = 0;
	cudaMemset(pair.d_data, 0, size*sizeof(int));
	pair._size = size;

	return pair;
}

DualDataf DualData_CreateFloat(unsigned int size){
	DualDataf pair;

	pair.h_data = (float*)malloc(size*sizeof(float));
	cudaError_t err = cudaMalloc((void**)&pair.d_data, size*sizeof(float));
	manageError(err);

	for(unsigned int i=0; i < size; ++i)
		pair.h_data[i] = 0;
	cudaMemset(pair.d_data, 0, size*sizeof(float));
	pair._size = size;

	return pair;
}

void DualData_CopyToHost(DualDatai data){
	cudaError_t err = cudaMemcpy(data.h_data, data.d_data, data._size*sizeof(int), cudaMemcpyDeviceToHost);
	manageError(err);
}

void DualData_CopyToDevice(DualDatai data){
	cudaError_t err = cudaMemcpy(data.d_data, data.h_data, data._size*sizeof(int), cudaMemcpyHostToDevice);
	manageError(err);
}

void DualData_CopyToHost(DualDataf data){
	cudaError_t err = cudaMemcpy(data.h_data, data.d_data, data._size*sizeof(float), cudaMemcpyDeviceToHost);
	manageError(err);
}

void DualData_CopyToDevice(DualDataf data){
	cudaError_t err = cudaMemcpy(data.d_data, data.h_data, data._size*sizeof(float), cudaMemcpyHostToDevice);
	manageError(err);
}

void DualData_Destroy(DualDatai data){
	free(data.h_data);		data.h_data = 0;
	cudaFree(data.d_data);	data.d_data = 0;
}

void DualData_Destroy(DualDataf data){
	free(data.h_data);		data.h_data = 0;
	cudaFree(data.d_data);	data.d_data = 0;
}


