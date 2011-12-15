/*
 * Chi2LibcuHighDensity.h
 *
 *  Created on: 13/12/2011
 *      Author: juanin
 */
#include "Container/cuMyMatrix.h"
#include "Container/cuMyPeak.h"

#ifndef CHI2LIBCUHIGHDENSITY_H_
#define CHI2LIBCUHIGHDENSITY_H_

class Chi2LibcuHighDensity {
public:
	/**
	 * Escala la Imagen elevandola al cuadrado y multiplicandola por 4.
	 */
	static void scaleImage(cuMyMatrix* img, cuMyMatrix* out);

	/**
	 * Invierte la imagen mediante el valor maximo.
	 */
	static void invertImage(cuMyMatrix* img, float maxval);

	/**
	 * Chequea si los Peaks encontrados se ubican en el interior de la imagen y los agrega al antiguo arreglo.
	 */
	static unsigned int checkInsidePeaks(cuMyPeakArray *old_peaks, cuMyPeakArray *new_peaks, cuMyMatrix *img, unsigned int os);

	/**
	 * Filtra los Peaks que se encuentran afuera de la imagen.
	 */
	static void filterPeaksOutside(cuMyPeakArray *peaks, cuMyMatrix *img, unsigned int os);

	/**
	 * Encuantra los parametros Mu y Sigma para un ajuste gausiano
	 */
	static std::pair<double, double> gaussianFit(cuMyPeakArray *peaks, cuMyMatrix *img, unsigned int ss);
};


#endif /* CHI2LIBCUHIGHDENSITY_H_ */
