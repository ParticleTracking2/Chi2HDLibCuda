/*
 * Chi2HD_Cuda.h
 *
 *  Created on: 15/11/2011
 *      Author: juanin
 */

#ifndef CHI2HD_CUDA_H_
#define CHI2HD_CUDA_H_

/**
 * Mini Array en cuda
 */
extern "C"
struct cuMyArray2D{
	float * array;
	unsigned int _sizeX;
	unsigned int _sizeY;

	unsigned int getSize(){
		return _sizeX*-_sizeY;
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

/**
 * Crea un arreglo en memoria de dispositivo.
 */
extern "C" cuMyArray2D CHI2HD_createArray(unsigned int sx, unsigned int sy);

/**
 * Elimina un arreglo en memoria de dispositivo.
 */
extern "C" void CHI2HD_destroyArray(cuMyArray2D *arr);

/**
 * Eleva al cuadrado cada elemento del arreglo en memoria del dispositivo.
 * @param arr
 */
extern "C" void CHI2HD_squareIt(float* arr, unsigned int size);


#endif
