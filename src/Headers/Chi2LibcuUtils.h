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

/*! \mainpage Biblioteca Chi2HD CUDA C++
 *
 * \section intro_sec Introdución
 *
 * Esta biblioteca esta pensada para ser usada con PTrack2 para poder acelerar los calculos mediante uso de GPU con tecnologia CUDA.<br/>
 * Esta página contiene todo lo necesario para entender cada una de las funciones y relaciones que se usan en la biblioteca. <br/>
 * Para más información acerca del uso, instalacion y recursos necesarios visita el <a href="http://trac.assembla.com/particle-tracking-2">Trac</a>
 *
 */

#if defined(__cplusplus)
extern "C++" {
#endif

/**
 * Encuentra el tamaño optimo de la grilla en 1D para lanzar un kernel en CUDA.
 * @param size tamaño de los elementos a calcular.
 * @return tamaño optimo de la grilla.
 */
unsigned int _findOptimalGridSize(unsigned int size);

/**
 * Encuentra el tamaño optimo de la grilla en 2D para lanzar un kernel en CUDA.
 * @param sizeX Tamaño de los elementos a calcular en la dimension X
 * @param sizeY Tamaño de los elementos a calcular en la dimension Y
 * @return Dimension optima de la grilla en X e Y.
 */
std::pair<unsigned int, unsigned int> _findOptimalGridSize(unsigned int sizeX, unsigned int sizeY);

/**
 * Encuentra el tamaño optimo de los bloques en 1D para lanzar un kernel en CUDA.
 * @param size tamaño de los elementos a calcular.
 * @return tamaño optimo de cada bloque.
 */
unsigned int _findOptimalBlockSize(unsigned int size);

/**
 * Encuentra el tamaño optimo de los bloques en 2D para lanzar un kernel en CUDA.
 * @param sizeX Tamaño de los elementos a calcular en la dimension X
 * @param sizeY Tamaño de los elementos a calcular en la dimension Y
 * @return Dimension optima de cada bloque en X e Y.
 */
std::pair<unsigned int, unsigned int> _findOptimalBlockSize(unsigned int sizeX, unsigned int sizeY);

/**
 * Maneja los errores de CUDA, ya sea de ejecucion, memoria y otros.
 * Si existe algun error se imprime en pantalla y se sale del programa con codigo -1.
 * @param err Tipo de error de ejecucion de CUDA.
 */
void manageError(cudaError_t err);

/**
 * Chequea los errores de lanzamiento de los kernel y sincroniza.
 */
void checkAndSync();

#if defined(__cplusplus)
}
#endif

#endif
