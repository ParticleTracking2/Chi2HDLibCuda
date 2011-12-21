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

struct DeviceProps{
	char name[256];
	int device;
};

class Chi2Libcu {
public:
	/**
	 * Establece el dispositivo que se ocupara en todas las futuras llamadas
	 */
	static void setDevice(int device);

	/**
	 * Retorna el nombre del dispositivo seleccionado
	 */
	static DeviceProps getProps();

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
	 * Valida que los peaks tengan un minimo de separacion
	 */
	static void validatePeaks(cuMyPeakArray* peaks, int mindistance);

	/**
	 * Generar las matrices auxiliares
	 */
	static void generateGrid(cuMyPeakArray* peaks, unsigned int shift, cuMyMatrix* grid_x, cuMyMatrix* grid_y, cuMyMatrixi* over);

	/**
	 * Calcula la diferencia con la Imagen Chi2 y la Imagen normal
	 */
	static float computeDifference(cuMyMatrix *img,cuMyMatrix *grid_x, cuMyMatrix *grid_y, float d, float w, cuMyMatrix *diffout);

	/**
	 * Trata de mejorar el centro de las particulas mediante el metodo de newton
	 */
	static void newtonCenter(cuMyMatrixi *over, cuMyMatrix *diff, cuMyPeakArray *peaks, int shift, float D, float w, float dp, float maxdr = 20.0);

};

#endif /* CHI2LIBCU_H_ */
