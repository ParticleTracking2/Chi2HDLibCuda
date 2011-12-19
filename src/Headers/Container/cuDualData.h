/*
 * cuDualData.h
 *
 *  Created on: 12/12/2011
 *      Author: juanin
 */
#include "../Chi2LibcuUtils.h"

#ifndef CUDUALDATA_H_
#define CUDUALDATA_H_

template <class DataType>
class DualData{
private:
	DataType* h_data;
	DataType* d_data;
	unsigned int _size;
	void allocateDevice();
	void allocateHost();
public:
	DualData();
	DualData(unsigned int size);
	DualData(unsigned int size, DataType def);
	~DualData();

	void reset(DataType def = 0);
	unsigned int size();
	void copyToHost();
	void copyToDevice();

	DataType* devicePointer();
	DataType* hostPointer();
	DataType & operator [](unsigned int index);
};

/*******************************
 * Constructores y Destructores
 *******************************/
template <class DataType>
DualData<DataType>::DualData(){
	_size = 1;
	allocateDevice();
	allocateHost();
	reset(0);
}

template <class DataType>
DualData<DataType>::DualData(unsigned int size){
	_size = size;
	allocateDevice();
	allocateHost();
}

template <class DataType>
DualData<DataType>::DualData(unsigned int size, DataType def){
	_size = size;
	allocateDevice();
	allocateHost();
	reset(def);
}

template <class DataType>
DualData<DataType>::~DualData(){
	free(h_data);
	cudaFree(d_data);
	_size = 0;
	h_data = 0;
	d_data = 0;
}

template <class DataType>
void DualData<DataType>::allocateDevice(){
	cudaError_t err = cudaMalloc((void**)&d_data, _size*sizeof(DataType));
	manageError(err);
}

template <class DataType>
void DualData<DataType>::allocateHost(){
	h_data = (DataType*)malloc(_size*sizeof(DataType));
}

/*******************************
 * Metodos
 *******************************/
template <class DataType>
void DualData<DataType>::reset(DataType def){
	for(unsigned int i=0; i < _size; ++i)
		h_data[i] = def;
	copyToDevice();
}

template <class DataType>
unsigned int DualData<DataType>::size(){
	return _size;
}

template <class DataType>
void DualData<DataType>::copyToHost(){
	cudaError_t err = cudaMemcpy(h_data, d_data, _size*sizeof(DataType), cudaMemcpyDeviceToHost);
	manageError(err);
}

template <class DataType>
void DualData<DataType>::copyToDevice(){
	cudaError_t err = cudaMemcpy(d_data, h_data, _size*sizeof(DataType), cudaMemcpyHostToDevice);
	manageError(err);
}

template <class DataType>
DataType* DualData<DataType>::devicePointer(){
	return d_data;
}

template <class DataType>
DataType* DualData<DataType>::hostPointer(){
	return h_data;
}

template <class DataType>
DataType & DualData<DataType>::operator [](unsigned int index){
	return h_data[index];
}

/*******************************
 * Fin Clase
 *******************************/
template <class DataType>
struct DualDataStruct{
	DataType* d_data;
	DataType* h_data;
	unsigned int _size;
};

#endif /* CUDUALDATA_H_ */
