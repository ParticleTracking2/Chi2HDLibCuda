/*
 * Chi2LibcuUtils.h
 *
 *  Created on: 23/11/2011
 *      Author: juanin
 */

#ifndef CHI2LIBCUUTILS_H_
#define CHI2LIBCUUTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <utility>

#if defined(__cplusplus)
extern "C++" {
#endif

/**
 * Encuentra el tamaño optimo de la grilla en 1D
 */
unsigned int _findOptimalGridSize(unsigned int size);

/**
 * Encuentra el tamaño optimo de la grilla en 2D
 */
std::pair<unsigned int, unsigned int> _findOptimalGridSize(unsigned int sizeX, unsigned int sizeY);

/**
 * Encuentra el tamaño optimo de los bloques en 1D
 */
unsigned int _findOptimalBlockSize(unsigned int size);

/**
 * Encuentra el tamaño optimo de los bloques en 2D
 */
std::pair<unsigned int, unsigned int> _findOptimalBlockSize(unsigned int sizeX, unsigned int sizeY);

/**
 * Maneja los errores de CUDA
 */
void manageError(cudaError_t err);

/**
 * Chequea los errores de lanzamiento de los kernel y sincroniza
 */
void checkAndSync();

#if defined(__cplusplus)
}
#endif

#endif
