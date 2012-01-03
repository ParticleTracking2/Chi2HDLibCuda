/*
 * cuDualData.h
 *
 *  Created on: 12/12/2011
 *      Author: juanin
 */
#include "../Chi2LibcuUtils.h"

#ifndef CUDUALDATA_H_
#define CUDUALDATA_H_

/**
 * Clase para contener arreglos de datos de cualquier tipo tanto en Dispositivo como en Host.
 * Esta clase debe compilarse con nvcc.
 */
template <class DataType>
class DualData{
private:

	/**
	 * Puntero a un Arreglo de datos dentro del host.
	 */
	DataType* h_data;

	/**
	 * Puntero a un Arreglo de datos dentro del dispositivo.
	 */
	DataType* d_data;

	/**
	 * Tamaño del contenedor
	 */
	unsigned int _size;

	/**
	 * Reserva memoria en dispositivo del tamaño de _size * tamaño del tipo de datos usado.
	 */
	void allocateDevice();

	/**
	 * Reserva memoria en Host del tamaño de _size * tamaño del tipo de datos usado.
	 */
	void allocateHost();
public:

	/**
	 * Constructor vacio, reserva espacio de 1 solo elemento.
	 */
	DualData();

	/**
	 * Construye el contenedor con tamaño igual a size.
	 * @param size Tamaño del contenedor.
	 */
	DualData(unsigned int size);

	/**
	 * Construye el contenedor con tamaño igual a size.
	 * Deja por defecto los valores del contenido como def.
	 * @param size Tamaño del contenedor.
	 * @param def Valores por defecto del contenido.
	 */
	DualData(unsigned int size, DataType def);

	/**
	 * Destructor de la clase.
	 * Libera memoria en Host y en Dispositivo dejando el tamaño del contenedor en 0.
	 */
	~DualData();

	/**
	 * Establece el valor de cada elemento dentro del contenedor como def.
	 * @param def Valor de cada elemento a ser asignado.
	 */
	void reset(DataType def = 0);

	/**
	 * Devuelve el tamaño del contenedor.
	 * @return Tamaño del contenedor.
	 */
	unsigned int size();

	/**
	 * Copia el contenido del dispositivo al host.
	 */
	void copyToHost();

	/**
	 * Copia el contenido del host al dispositivo.
	 */
	void copyToDevice();

	/**
	 * Devuelve el puntero a los datos dentro del dispositivo
	 * @return Puntero a datos del dispositivo.
	 */
	DataType* devicePointer();

	/**
	 * Devuelve el puntero a los datos dentro del host
	 * @return Puntero a datos del host.
	 */
	DataType* hostPointer();

	/**
	 * Operador [] para acceder a los contenidos del host.
	 * Estos valores son modificables
	 * @return Contenido del indice index dentro del contenedor.
	 */
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
