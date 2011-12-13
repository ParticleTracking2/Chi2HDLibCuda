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
	if(_size > 0){
		cudaError_t err = cudaMalloc((void**)&_device_array, (size_t)(_size*sizeof(cuMyPeak)));
		manageError(err);
	}
}

void cuMyPeakArray::allocateHost(){
	if(_size > 0)
		_host_array = (cuMyPeak*)malloc(_size*sizeof(cuMyPeak));
}

cuMyPeakArray::~cuMyPeakArray(){
	deallocateDevice();
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
