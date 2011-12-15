/*
 * cuMyPeakArray.cu
 *
 *  Created on: 11/12/2011
 *      Author: juanin
 */

#include "../Headers/Container/cuMyPeak.h"
#include "../Headers/Chi2LibcuUtils.h"
#include <iostream>

void cuMyPeakArray::goEmpty(){
	_host_array = 0;
	_device_array = 0;
	_size = 0;
}

/**
 *******************************
 * Constructores y Destructores
 *******************************
 */
cuMyPeakArray::cuMyPeakArray(){
	goEmpty();
}

cuMyPeakArray::cuMyPeakArray(unsigned int size){
	goEmpty();
	_size = size;
	allocateDevice();
}

void cuMyPeakArray::allocateDevice(){
	if(_size > 0 && !_device_array){
		cudaError_t err = cudaMalloc((void**)&_device_array, (size_t)(_size*sizeof(cuMyPeak)));
		manageError(err);
	}
}

void cuMyPeakArray::allocateHost(){
	if(_size > 0 && !_host_array)
		_host_array = (cuMyPeak*)malloc(_size*sizeof(cuMyPeak));
}

cuMyPeakArray::~cuMyPeakArray(){
	deallocateDevice();
}

void cuMyPeakArray::deallocate(){
	deallocateDevice();
	deallocateHost();
	_size = 0;
}

void cuMyPeakArray::deallocateDevice(){
	if(_device_array){
		cudaError_t err = cudaFree(_device_array);
		manageError(err);
	}
	_device_array = 0;
	if(!_host_array)
		_size = 0;
}

void cuMyPeakArray::deallocateHost(){
	if(_host_array){
		free(_host_array);
	}
	_host_array = 0;
	if(!_device_array)
		_size = 0;
}

/**
 *******************************
 * Metodos
 *******************************
 */

void cuMyPeakArray::append(cuMyPeakArray* data){
	unsigned int new_size = _size+data->size();

	_host_array = (cuMyPeak*)malloc(new_size*sizeof(cuMyPeak));
	copyToHost(); deallocateDevice();
	data->copyToHost();

	cuMyPeak* ptr = data->hostPointer();
	for(unsigned int index=_size; index < new_size; ++index){
		_host_array[index] = ptr[index-_size];
	}
	ptr = 0;
	data->deallocateHost();
	_size = new_size;
	allocateDevice();
	copyToDevice();
	deallocateHost();
}

__global__ void __includeDeltas(cuMyPeak* arr, unsigned int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= size)
		return;

	arr[idx].fx = arr[idx].fx + arr[idx].dfx;
	arr[idx].fy = arr[idx].fy + arr[idx].dfy;

	arr[idx].x = (int)rintf(arr[idx].fx);
	arr[idx].y = (int)rintf(arr[idx].fy);
}

void cuMyPeakArray::includeDeltas(){
	dim3 dimGrid(_findOptimalBlockSize(_size));
	dim3 dimBlock(_findOptimalBlockSize(_size));
	__includeDeltas<<<dimGrid, dimBlock>>>(_device_array, _size);
	cudaError_t err = cudaDeviceSynchronize();
	manageError(err);
}

struct internal_cuMyPeakCompareChi {
  __host__ __device__
  bool operator()(const cuMyPeak &lhs, const cuMyPeak &rhs){
	  return lhs.chi_intensity < rhs.chi_intensity;
  }
};

void cuMyPeakArray::sortByChiIntensity(){
	thrust::device_vector<cuMyPeak> dv = deviceVector();
	thrust::stable_sort(dv.begin(), dv.end(), internal_cuMyPeakCompareChi());
	deviceVector(dv);
}

void cuMyPeakArray::copyToHost(){
	if(!_host_array){
		allocateHost();
	}
	cudaError_t err = cudaMemcpy(_host_array, _device_array, _size*sizeof(cuMyPeak), cudaMemcpyDeviceToHost);
	manageError(err);
}

void cuMyPeakArray::copyToDevice(){
	if(!_device_array){
		allocateDevice();
	}
	cudaError_t err = cudaMemcpy(_device_array, _host_array, _size*sizeof(cuMyPeak), cudaMemcpyHostToDevice);
	manageError(err);
}

unsigned int cuMyPeakArray::size(){
	return _size;
}

cuMyPeak* cuMyPeakArray::devicePointer(){
	return _device_array;
}

cuMyPeak* cuMyPeakArray::hostPointer(){
	return _host_array;
}

void cuMyPeakArray::keepValids(){
	//Contar Validos
	copyToHost();
	cuMyPeak tmp[_size];

	unsigned int valids = 0;
	for(unsigned int i=0; i < _size; ++i){
		if(_host_array[i].valid){
			tmp[valids] = _host_array[i];
			++valids;
		}
	}

	// Borrar datos y copiar
	deallocateDevice();	deallocateHost();
	_size = valids;
	allocateHost();
	for(unsigned int i=0; i < _size; ++i){
		_host_array[i] = tmp[i];
	}
	copyToDevice();
}

thrust::device_vector<cuMyPeak> cuMyPeakArray::deviceVector(){
	thrust::host_vector<cuMyPeak> hv(_size);
	copyToHost();
	for(unsigned int i=0; i < _size; ++i){
		hv[i] = _host_array[i];
	}
	thrust::device_vector<cuMyPeak> ret = hv;
	return ret;
}
void cuMyPeakArray::deviceVector(thrust::device_vector<cuMyPeak> dv){
	thrust::host_vector<cuMyPeak> hv = dv;
	for(unsigned int i=0; i < _size; ++i){
		_host_array[i] = hv[i];
	}
	copyToDevice();
}

cuMyPeak cuMyPeakArray::getHostValue(unsigned int index){
	return _host_array[index];
}

cuMyPeak & cuMyPeakArray::atHost(unsigned int index){
	return _host_array[index];
}
