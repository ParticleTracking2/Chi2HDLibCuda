/*
 * cuDualData.cu
 *
 *  Created on: 12/12/2011
 *      Author: juanin
 */

#include "../Headers/Container/cuDualData.h"
#include "../Headers/Chi2LibcuUtils.h"

DualDatai DualData_CreateInt(){
	DualDatai pair;

	pair.h_data = (int*)malloc(sizeof(int));
	cudaError_t err = cudaMalloc((void**)&pair.d_data, sizeof(int));
	manageError(err);

	pair.h_data[0] = 0;
	cudaMemset(pair.d_data, 0, sizeof(int));

	return pair;
}

DualDataf DualData_CreateFloat(){
	DualDataf pair;

	pair.h_data = (float*)malloc(sizeof(float));
	cudaError_t err = cudaMalloc((void**)&pair.d_data, sizeof(float));
	manageError(err);

	pair.h_data[0] = 0;
	cudaMemset(pair.d_data, 0, sizeof(float));

	return pair;
}

void DualData_CopyToHost(DualDatai data){
	cudaError_t err = cudaMemcpy(data.h_data, data.d_data, sizeof(int), cudaMemcpyDeviceToHost);
	manageError(err);
}

void DualData_CopyToDevice(DualDatai data){
	cudaError_t err = cudaMemcpy(data.d_data, data.h_data, sizeof(int), cudaMemcpyHostToDevice);
	manageError(err);
}

void DualData_CopyToHost(DualDataf data){
	cudaError_t err = cudaMemcpy(data.h_data, data.d_data, sizeof(float), cudaMemcpyDeviceToHost);
	manageError(err);
}

void DualData_CopyToDevice(DualDataf data){
	cudaError_t err = cudaMemcpy(data.d_data, data.h_data, sizeof(float), cudaMemcpyHostToDevice);
	manageError(err);
}

void DualData_Destroy(DualDatai data){
	free(data.h_data);
	cudaFree(data.d_data);
}

void DualData_Destroy(DualDataf data){
	free(data.h_data);
	cudaFree(data.d_data);
}


