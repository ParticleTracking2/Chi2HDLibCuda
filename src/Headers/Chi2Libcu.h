/*
 * Chi2libcu.h
 *
 *  Created on: 10/12/2011
 *      Author: juanin
 */

#include "Container/cuMyMatrix.h"
#include "Container/cuMyPeak.h"
#include <utility>

#ifndef CHI2LIBCU_H_
#define CHI2LIBCU_H_

using namespace std;

class Chi2Libcu {
public:
	/**
	 * Obtiene los valores maximos y minimos de un arreglo
	 */
	static pair<float, float> minMax(cuMyMatrix *arr);

	/**
	 * Normalizar Imagen
	 */
	static void normalize(cuMyMatrix *arr, float _min, float _max);

	/**
	 * Genera un kernel en GPU y lo copia al Host tambien.
	 */
	static cuMyMatrix gen_kernel(unsigned int ss, unsigned int os, float d, float w);

	/**
	 * Obtener Peaks
	 */
	static cuMyPeakArray getPeaks(cuMyMatrix *arr, int threshold, int mindistance, int minsep);

	/**
	 * Generar las matrices auxiliares
	 */
	static void generateGrid(cuMyPeakArray* peaks, unsigned int shift, cuMyMatrix* grid_x, cuMyMatrix* grid_y, cuMyMatrixi* over);
};

#endif /* CHI2LIBCU_H_ */
