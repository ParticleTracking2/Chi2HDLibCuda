/*
 * Chi2HD_Cuda.h
 *
 *  Created on: 15/11/2011
 *      Author: juanin
 */

#ifndef CHI2HD_CUDA_H_
#define CHI2HD_CUDA_H_
#include <stdlib.h>
#include "Chi2HD_CudaStructs.h"

#if defined(__cplusplus)
extern "C" {
#endif
/**
 * Crea un arreglo en memoria de dispositivo.
 */
cuMyArray2D CHI2HD_createArray(unsigned int sx, unsigned int sy);

/**
 * Crea un arreglo en memoria de dispositivo y retorna un puntero.
 */
void CHI2HD_createArrayPointer(unsigned int sx, unsigned int sy, cuMyArray2D* ret);

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
 * Copia el arreglo completamente de la fuente al destino
 */
void CHI2HD_copy(cuMyArray2D *src, cuMyArray2D *dst);

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

/**
 * Genera un kernel en GPU y lo copia al Host tambien.
 */
cuMyArray2D CHI2HD_gen_kernel(unsigned int ss, unsigned int os, float d, float w);

/**
 * Obtener Peaks
 */
void CHI2HD_getPeaks(cuMyArray2D *arr, int threshold, int mindistance, int minsep, cuPeak* peaks);

#if defined(__cplusplus)
}
#endif

#endif
