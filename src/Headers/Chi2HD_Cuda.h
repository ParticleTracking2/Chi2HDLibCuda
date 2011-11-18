/*
 * Chi2HD_Cuda.h
 *
 *  Created on: 15/11/2011
 *      Author: juanin
 */

#ifndef CHI2HD_CUDA_H_
#define CHI2HD_CUDA_H_
#include <stdlib.h>

/**
 * Mini Array en cuda
 */
struct cuMyArray2D{
	unsigned int _sizeX;
	unsigned int _sizeY;

	float * _device_array;
	float * _host_array;
	int _device;

	unsigned int getSize(){
		return _sizeX*_sizeY;
	}

	// Cuidado con esta operacion, si tiene mal los parametros...
	unsigned int position(unsigned int x, unsigned int y){
		return _x+_sizeY*_y;
	}

	unsigned int safePosition(unsigned int x, unsigned int y){
		unsigned int _x = x%_sizeX;
		unsigned int _y = y%_sizeY;
		return _x+_sizeY*_y;
	}

	float getValueHost(unsigned int position){
		return _host_array[position];
	}
};

struct myPair{
	float first;
	float second;
};

#if defined(__cplusplus)
extern "C" {
#endif
/**
 * Crea un arreglo en memoria de dispositivo.
 */
cuMyArray2D CHI2HD_createArray(unsigned int sx, unsigned int sy);

/**
 * Elimina un arreglo en memoria de dispositivo.
 */
bool CHI2HD_destroyArray(cuMyArray2D *arr);

/**
 * Reinicia el arreglo a o por defecto o al numero que se le asigne.
 */
void CHI2HD_reset(cuMyArray2D *arr, float def = 0);

/**
 * Eleva al cuadrado cada elemento del arreglo en memoria del dispositivo.
 * @param arr
 */
void CHI2HD_squareIt(cuMyArray2D *arr);

/**
 * Eleva al cubo cada elemento del arreglo en memoria del dispositivo.
 * @param arr
 */
void CHI2HD_cubeIt(cuMyArray2D *arr);

/**
 * Copia el arreglo del dispositivo a memoria del host, dejando los punteros en la estructura dada
 */
void CHI2HD_copyToHost(cuMyArray2D *arr);

/**
 * Copia el arreglo de memoria en host a dispositivo, dejando los punteros en la estructura dada
 */
void CHI2HD_copyToDevice(cuMyArray2D *arr);

/**
 * Obtiene los valores maximos y minimos de un arreglo
 */
myPair CHI2HD_minMax(cuMyArray2D *arr);

/**
 * Normalizar Imagen
 */
void CHI2HD_normalize(cuMyArray2D *arr, float _min, float _max);

#if defined(__cplusplus)
}
#endif

#endif
