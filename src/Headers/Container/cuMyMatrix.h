/*
 * cuMyMatrix.h
 *
 *  Created on: 07/12/2011
 *      Author: juanin
 */

#ifndef CUMYMATRIX_
#define CUMYMATRIX_
#include <stdlib.h>

using namespace std;

/**
 * Matriz de punto flotante de datos en 2 Dimensiones mapeada en 1 Dimension para utilizar en CUDA y CPU.
 */
class cuMyMatrix{
private:

	/**
	 * Puntero a un arreglo en dispositivo.
	 */
	float * _device_array;

	/**
	 * Puntero a un arreglo en CPU.
	 */
	float * _host_array;

	/**
	 * Indicador de residencia de la memoria en dispositivo.
	 * Actualmente no se esta utilizando.
	 */
	int device;

	/**
	 * Dimension de la Matriz en X.
	 */
	unsigned int _sizeX;

	/**
	 * Dimension de la Matriz en Y.
	 */
	unsigned int _sizeY;

	/**
	 * Reserva memoria en Dispositivo, el tamaño esta indicado por _sizeY*_sizeX.
	 */
	void allocateDevice();

	/**
	 * Reserva memoria en HOST, el tamaño esta indicado por _sizeY*_sizeX.
	 */
	void allocateHost();

	/**
	 * Establece todos los elementos de la matriz a 0.
	 */
	void goEmpty();
public:
	/**
	 * Constructor vacio.
	 * Se debe llamar a allocate(unsigned int, unsigned int) para que la matriz quede usable.
	 */
	cuMyMatrix();

	/**
	 * Crea una Matriz de floats vacia.
	 * @param x Tamaño de la Matriz en X.
	 * @param y Tamaño de la Matriz en Y.
	 */
	cuMyMatrix(unsigned int x, unsigned int y);

	/**
	 * Crea una Matriz de floats con todos los elementos establecidos como def.
	 * @param x Tamaño de la Matriz en X.
	 * @param y Tamaño de la Matriz en Y.
	 * @param def Valor por defecto de todas las celdas.
	 */
	cuMyMatrix(unsigned int x, unsigned int y, float def);

	/**
	 * Destructor de la Matriz.
	 * Libera los recursos del dispositivo y de HOST.
	 */
	~cuMyMatrix();

	/**
	 * Reserva memoria tanto en el dispositivo como en HOST.
	 * El tamaño de la matriz cambia a x*y.
	 * @param x Tamaño de la Matriz en X.
	 * @param y Tamaño de la Matriz en Y.
	 */
	void allocate(unsigned int x, unsigned int y);

	/**
	 * Reserva memoria en el dispositivo.
	 * El tamaño de la matriz cambia a x*y.
	 * @param x Tamaño de la Matriz en X.
	 * @param y Tamaño de la Matriz en Y.
	 */
	void allocateDevice(unsigned int x, unsigned int y);

	/**
	 * Reserva memoria en el HOST.
	 * El tamaño de la matriz cambia a x*y.
	 * @param x Tamaño de la Matriz en X.
	 * @param y Tamaño de la Matriz en Y.
	 */
	void allocateHost(unsigned int x, unsigned int y);

	/**
	 * Libera memoria tanto en Dispositivo como en Host.
	 */
	void deallocate();

	/**
	 * Libera memoria en Host.
	 */
	void deallocateHost();

	/**
	 * Libera memoria en Dispositivo.
	 */
	void deallocateDevice();

	/**
	 * Retorna el tamaño real de la Matriz, esto es Tamaño en X * Tamaño en Y.
	 * @return Tamaño real de la Matriz.
	 */
	unsigned int size();

	/**
	 * Retorna el tamaño de la matriz en X.
	 * @Return Tamaño en X de la Matriz.
	 */
	unsigned int sizeX();

	/**
	 * Retorna el tamaño de la matriz en Y.
	 * @Return Tamaño en Y de la Matriz.
	 */
	unsigned int sizeY();

	/**
	 * Copia los valores que se encuentran en memoria del HOST al dispositivo.
	 * Si no se encuentra la memoria en dispositivo reservada, se reserva.
	 */
	void copyToDevice();

	/**
	 * Copia los valores que se encuentran en memoria del dispositivo al HOST.
	 * Si no se encuentra la memoria en HOST reservada, se reserva.
	 */
	void copyToHost();

	/**
	 * Establece todos los valores de las celdas de la matriz en Dispositivo segun def.
	 * Si no se especifica, se establecen a 0.
	 * @param def valor por defecto de las celdas.
	 */
	void reset(float def = 0);

	/**
	 * Retorna el puntero a los datos del Dispositivo.
	 * No se verifica la existencia de estos.
	 * @see copyToDevice()
	 * @see deallocateDevice()
	 * @return puntero a datos del dispositivo.
	 */
	float* devicePointer();

	/**
	 * Retorna el puntero a los datos del HOST.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @return puntero a datos del HOST.
	 */
	float* hostPointer();

	/**
	 * Obtiene el valor de la celda de la Matriz mapeada a 1 Dimension dentro del HOST.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @param index indice dentro del arreglo.
	 * @return valor de la celda indicada por index.
	 */
	float getValueHost(unsigned int index);

	/**
	 * Obtiene el valor de la celda de la Matriz dentro del HOST.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @param x indice en x dentro de la matriz.
	 * @param y indice en y dentro de la matriz.
	 * @return valor de la celda indicada por x e y.
	 */
	float getValueHost(unsigned int x, unsigned int y);

	/**
	 * Obtiene el contenido de la celda de la Matriz dentro del HOST dentro del HOST.
	 * Este valor puede ser modificado.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @param x indice en x dentro de la matriz.
	 * @param y indice en y dentro de la matriz.
	 * @return contenido de la celda indicada por index.
	 */
	float & atHost(unsigned int x, unsigned int y);

	/**
	 * Obtiene el contenido de la celda de la Matriz mapeada a 1 Dimension dentro del HOST.
	 * Este valor puede ser modificado.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @param index indice dentro del arreglo.
	 * @return contenido de la celda indicada por x e y.
	 */
	float & atHost(unsigned int index);

	/**
	 * Operador = para copiar cuMyMatrix
	 * @param mtrx cuMyMatrix a copiar
	 */
	void operator = (cuMyMatrix mtrx);
};

/**
 * Matriz de enteros de datos en 2 Dimensiones mapeada en 1 Dimension para utilizar en CUDA y CPU.
 */
class cuMyMatrixi{
private:

	/**
	 * Puntero a un arreglo en dispositivo.
	 */
	int * _device_array;

	/**
	 * Puntero a un arreglo en CPU.
	 */
	int * _host_array;

	/**
	 * Indicador de residencia de la memoria en dispositivo.
	 * Actualmente no se esta utilizando.
	 */
	int device;

	/**
	 * Dimension de la Matriz en X.
	 */
	unsigned int _sizeX;

	/**
	 * Dimension de la Matriz en Y.
	 */
	unsigned int _sizeY;

	/**
	 * Reserva memoria en Dispositivo, el tamaño esta indicado por _sizeY*_sizeX.
	 */
	void allocateDevice();

	/**
	 * Reserva memoria en HOST, el tamaño esta indicado por _sizeY*_sizeX.
	 */
	void allocateHost();

	/**
	 * Establece todos los elementos de la matriz a 0.
	 */
	void goEmpty();
public:

	/**
	 * Constructor vacio.
	 * Se debe llamar a allocate(unsigned int, unsigned int) para que la matriz quede usable.
	 */
	cuMyMatrixi();

	/**
	 * Crea una Matriz de enteros vacia.
	 * @param x Tamaño de la Matriz en X.
	 * @param y Tamaño de la Matriz en Y.
	 */
	cuMyMatrixi(unsigned int x, unsigned int y);

	/**
	 * Crea una Matriz de enteros con todos los elementos establecidos como def.
	 * @param x Tamaño de la Matriz en X.
	 * @param y Tamaño de la Matriz en Y.
	 * @param def Valor por defecto de todas las celdas.
	 */
	cuMyMatrixi(unsigned int x, unsigned int y, int def);

	/**
	 * Destructor de la Matriz.
	 * Libera los recursos del dispositivo y de HOST.
	 */
	~cuMyMatrixi();

	/**
	 * Reserva memoria tanto en el dispositivo como en HOST.
	 * El tamaño de la matriz cambia a x*y.
	 * @param x Tamaño de la Matriz en X.
	 * @param y Tamaño de la Matriz en Y.
	 */
	void allocate(unsigned int x, unsigned int y);

	/**
	 * Reserva memoria en el dispositivo.
	 * El tamaño de la matriz cambia a x*y.
	 * @param x Tamaño de la Matriz en X.
	 * @param y Tamaño de la Matriz en Y.
	 */
	void allocateDevice(unsigned int x, unsigned int y);

	/**
	 * Reserva memoria en el HOST.
	 * El tamaño de la matriz cambia a x*y.
	 * @param x Tamaño de la Matriz en X.
	 * @param y Tamaño de la Matriz en Y.
	 */
	void allocateHost(unsigned int x, unsigned int y);

	/**
	 * Libera memoria tanto en Dispositivo como en Host.
	 */
	void deallocate();

	/**
	 * Libera memoria en Host.
	 */
	void deallocateHost();

	/**
	 * Libera memoria en Dispositivo.
	 */
	void deallocateDevice();

	/**
	 * Retorna el tamaño real de la Matriz, esto es Tamaño en X * Tamaño en Y.
	 * @return Tamaño real de la Matriz.
	 */
	unsigned int size();

	/**
	 * Retorna el tamaño de la matriz en X.
	 * @Return Tamaño en X de la Matriz.
	 */
	unsigned int sizeX();

	/**
	 * Retorna el tamaño de la matriz en Y.
	 * @Return Tamaño en Y de la Matriz.
	 */
	unsigned int sizeY();

	/**
	 * Copia los valores que se encuentran en memoria del HOST al dispositivo.
	 * Si no se encuentra la memoria en dispositivo reservada, se reserva.
	 */
	void copyToDevice();

	/**
	 * Copia los valores que se encuentran en memoria del dispositivo al HOST.
	 * Si no se encuentra la memoria en HOST reservada, se reserva.
	 */
	void copyToHost();


	/**
	 * Establece todos los valores de las celdas de la matriz en Dispositivo segun def.
	 * Si no se especifica, se establecen a 0.
	 * @param def valor por defecto de las celdas.
	 */
	void reset(int def = 0);

	/**
	 * Retorna el puntero a los datos del Dispositivo.
	 * No se verifica la existencia de estos.
	 * @see copyToDevice()
	 * @see deallocateDevice()
	 * @return puntero a datos del dispositivo.
	 */
	int* devicePointer();

	/**
	 * Retorna el puntero a los datos del HOST.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @return puntero a datos del HOST.
	 */
	int* hostPointer();

	/**
	 * Obtiene el valor de la celda de la Matriz mapeada a 1 Dimension dentro del HOST.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @param index indice dentro del arreglo.
	 * @return valor de la celda indicada por index.
	 */
	int getValueHost(unsigned int index);

	/**
	 * Obtiene el valor de la celda de la Matriz dentro del HOST.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @param x indice en x dentro de la matriz.
	 * @param y indice en y dentro de la matriz.
	 * @return valor de la celda indicada por x e y.
	 */
	int getValueHost(unsigned int x, unsigned int y);

	/**
	 * Obtiene el contenido de la celda de la Matriz dentro del HOST dentro del HOST.
	 * Este valor puede ser modificado.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @param x indice en x dentro de la matriz.
	 * @param y indice en y dentro de la matriz.
	 * @return contenido de la celda indicada por index.
	 */
	int & atHost(unsigned int x, unsigned int y);

	/**
	 * Obtiene el contenido de la celda de la Matriz mapeada a 1 Dimension dentro del HOST.
	 * Este valor puede ser modificado.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @param index indice dentro del arreglo.
	 * @return contenido de la celda indicada por x e y.
	 */
	int & atHost(unsigned int index);

	/**
	 * Operador = para copiar cuMyMatrix
	 * @param mtrx cuMyMatrix a copiar
	 */
	void operator = (cuMyMatrixi mtrx);
};

#endif
