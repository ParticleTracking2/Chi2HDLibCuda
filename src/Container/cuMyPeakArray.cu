/*
 * cuMyPeakArray.cu
 *
 *  Created on: 11/12/2011
 *      Author: juanin
 */

#include "../Headers/Container/cuMyPeak.h"
#include "../Headers/Chi2LibcuUtils.h"

void cuMyPeakArray::goEmpty(){
	_host_array.clear();
	_device_array.clear();
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
	_device_array.reserve(_size);
}

void cuMyPeakArray::allocateHost(){
	_host_array.reserve(_size);
	cuMyPeak tmp;
	for(unsigned int i=0; i < _size; ++i){
		_host_array.push_back(tmp);
	}
}

cuMyPeakArray::~cuMyPeakArray(){
	goEmpty();
}

void cuMyPeakArray::deallocate(){
	deallocateDevice();
	deallocateHost();
	_size = 0;
}

void cuMyPeakArray::deallocateDevice(){
	_device_array.clear();
	if(_host_array.empty())
		_size = 0;
}

void cuMyPeakArray::deallocateHost(){
	_host_array.clear();
	if(_device_array.empty())
		_size = 0;
}

/**
 *******************************
 * Metodos
 *******************************
 */

void cuMyPeakArray::append(cuMyPeakArray* data){
	_device_array.resize(_size+data->size());
	cudaError_t err = cudaMemcpy(devicePointer()+_size, data->devicePointer(), data->size()*sizeof(cuMyPeak), cudaMemcpyDeviceToDevice);
	manageError(err);
	_size = _size+data->size();
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
	__includeDeltas<<<dimGrid, dimBlock>>>(_device_array.data().get(), _size);
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
	thrust::stable_sort(_device_array.begin(), _device_array.end(), internal_cuMyPeakCompareChi());
}

void cuMyPeakArray::copyToHost(){
	if(_host_array.empty()){
		allocateHost();
	}
	cudaError_t err = cudaMemcpy(hostPointer(), devicePointer(), _size*sizeof(cuMyPeak), cudaMemcpyDeviceToHost);
	manageError(err);
}

void cuMyPeakArray::copyToDevice(){
	if(_device_array.empty()){
		allocateDevice();
	}
	_device_array = _host_array;
}

unsigned int cuMyPeakArray::size(){
	return _size;
}

cuMyPeak* cuMyPeakArray::devicePointer(){
	return _device_array.data().get();
}

thrust::device_vector<cuMyPeak>* cuMyPeakArray::deviceVector(){
	return &_device_array;
}

cuMyPeak* cuMyPeakArray::hostPointer(){
	return &_host_array[0];
}

std::vector<cuMyPeak>* cuMyPeakArray::hostVector(){
	return &_host_array;
}

void cuMyPeakArray::keepValids(){
	//Contar Validos
	copyToHost();

	unsigned int valids = 0;
	for(unsigned int i=0; i < _host_array.size(); ++i){
		if(_host_array[i].valid){
			++valids;
		}else{
			_host_array.erase(_host_array.begin()+i);
			--i;
		}
	}

	_size = valids;
	// Borrar datos y copiar
	deallocateDevice();
	copyToDevice();
	deallocateHost();
}

cuMyPeak cuMyPeakArray::getHostValue(unsigned int index){
	return _host_array[index];
}

cuMyPeak & cuMyPeakArray::atHost(unsigned int index){
	return _host_array[index];
}
