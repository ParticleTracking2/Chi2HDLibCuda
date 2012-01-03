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

/**
 * Clase con funciones especificas para ejecutar el algoritmo de minimos cuadrados con alta densidad de particulas.
 * La mayoria de las funciones y estructuras de datos se ejecutan y residen en GPU.
 */
class Chi2LibcuHighDensity {
public:

	/**
	 * Escala la Matriz elevandola al cuadrado y multiplicandola por 4.
	 * @param img Matriz de Imagen base.
	 * @param out Matriz de imagen de salida.
	 */
	static void scaleImage(cuMyMatrix* img, cuMyMatrix* out);

	/**
	 * Invierte los valores de la Matriz mediante el valor maximo.
	 * @param img Matriz a invertir los valores.
	 * @param maxval valor maximo de la matriz.
	 */
	static void invertImage(cuMyMatrix* img, float maxval);

	/**
	 * Chequea si los Peaks encontrados se ubican en el interior de la imagen y los agrega al antiguo contenedor.
	 * @param old_peaks Contenedor de peaks antiguo en GPU.
	 * @param new_peaks Contenedor de peaks nuevos en GPU.
	 * @param img Matriz de Imagen original.
	 * @param os Tamaño /2 de la particula ideal.
	 * @return Cantidad de peaks nuevos agregados al contenedor antiguo.
	 */
	static unsigned int checkInsidePeaks(cuMyPeakArray *old_peaks, cuMyPeakArray *new_peaks, cuMyMatrix *img, unsigned int os);

	/**
	 * Filtra los Peaks que se encuentran afuera de la imagen original.
	 * @param peaks Contenedor de peaks en GPU.
	 * @param img Matriz de Imagen original.
	 * @param os Tamaño /2 de la particula ideal.
	 */
	static void filterPeaksOutside(cuMyPeakArray *peaks, cuMyMatrix *img, unsigned int os);

	/**
	 * Encuantra los parametros Mu y Sigma para un ajuste gausiano.
	 * @param peaks Contenedor de peaks en GPU.
	 * @param img Imagen original.
	 * @param ss Tamaño /2 de la particula ideal.
	 * @return Par de datos Mu y Sigma respectivamente.
	 */
	static std::pair<double, double> gaussianFit(cuMyPeakArray *peaks, cuMyMatrix *img, unsigned int ss);
};


#endif /* CHI2LIBCUHIGHDENSITY_H_ */
