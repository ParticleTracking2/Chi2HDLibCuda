/*
 * cuMyPeakArray.cu
 *
 *  Created on: 11/12/2011
 *      Author: juanin
 */

#include "../Headers/Container/cuMyPeak.h"
#include "../Headers/Chi2LibcuUtils.h"

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
	cudaError_t err = cudaMalloc((void**)&_device_array, _size*sizeof(cuMyPeak));
	manageError(err);
}

void cuMyPeakArray::allocateHost(){
	_host_array = (cuMyPeak*)malloc(_size*sizeof(cuMyPeak));
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
	if(_device_array)
		cudaFree(_device_array);
	_device_array = 0;
	if(_host_array == 0)
		_size = 0;
}

void cuMyPeakArray::deallocateHost(){
	if(_host_array)
		free(_host_array);
	_host_array = 0;
	if(_device_array == 0)
		_size = 0;
}

/**
 *******************************
 * Metodos
 *******************************
 */

void cuMyPeakArray::append(cuMyPeakArray* data){
	// Alocar nuevo tamaño
	cuMyPeak* newArray;
	cudaError_t err = cudaMalloc((void**)&newArray, (_size+data->size())*sizeof(cuMyPeak));
	manageError(err);

	// Copiar arreglo antiguo
	err = cudaMemcpy(newArray, _device_array, _size*sizeof(cuMyPeak), cudaMemcpyDeviceToDevice);
	manageError(err);

	// Copiar nuevo arreglo
	err = cudaMemcpy(newArray+_size, data->devicePointer(), data->size()*sizeof(cuMyPeak), cudaMemcpyDeviceToDevice);
	manageError(err);

	cudaFree(_device_array);
	_device_array = newArray;
	_size = _size+data->size();
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
	copyToHost();

	std::vector<cuMyPeak> tmp(_host_array, _host_array+_size);
	std::sort(tmp.begin(), tmp.end(), internal_cuMyPeakCompareChi());
	cudaError_t err = cudaMemcpy(_host_array, tmp.data(), _size*sizeof(cuMyPeak), cudaMemcpyHostToHost);
	manageError(err);

	copyToDevice();
}

struct internal_cuMyPeakCompareVor {
  __host__ __device__
  bool operator()(const cuMyPeak &lhs, const cuMyPeak &rhs){
	  return lhs.vor_area < rhs.vor_area;
  }
};

void cuMyPeakArray::sortByVoronoiArea(){
	copyToHost();

	std::vector<cuMyPeak> tmp(_host_array, _host_array+_size);
	std::sort(tmp.begin(), tmp.end(), internal_cuMyPeakCompareVor());
	cudaError_t err = cudaMemcpy(_host_array, tmp.data(), _size*sizeof(cuMyPeak), cudaMemcpyHostToHost);
	manageError(err);

	copyToDevice();
}

void cuMyPeakArray::copyToHost(){
	if(!_host_array){
		allocateHost();
	}
	cudaError_t err = cudaMemcpy(hostPointer(), devicePointer(), _size*sizeof(cuMyPeak), cudaMemcpyDeviceToHost);
	manageError(err);
}

void cuMyPeakArray::copyToDevice(){
	if(!_device_array){
		allocateDevice();
	}
	cudaError_t err = cudaMemcpy(devicePointer(), hostPointer(), _size*sizeof(cuMyPeak), cudaMemcpyHostToDevice);
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
			tmp[valids++] = _host_array[i];
		}
	}

	deallocateDevice(); deallocateHost();
	_size = valids;
	// Borrar datos y copiar
	allocateHost();
	memcpy(_host_array, tmp, _size*sizeof(cuMyPeak));
	copyToDevice();
}

cuMyPeak cuMyPeakArray::getHostValue(unsigned int index){
	return _host_array[index];
}

cuMyPeak & cuMyPeakArray::atHost(unsigned int index){
	return _host_array[index];
}
