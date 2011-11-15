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

	float * _array;
	float * _host_array;
	int _device;

	unsigned int getSize(){
		return _sizeX*_sizeY;
	}

	// Cuidado con esta operacion, si tiene mal los parametros...
	unsigned int position(unsigned int x, unsigned int y){
		return x*_sizeX+y;
	}

	unsigned int safePosition(unsigned int x, unsigned int y){
		unsigned int _x = x%_sizeX;
		unsigned int _y = y%_sizeY;
		return _x*_sizeX+_y;
	}
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
 * Copia el arreglo del dispositivo a memoria
 */
void CHI2HD_copyToHost(cuMyArray2D *arr);

#if defined(__cplusplus)
}
#endif

#endif
