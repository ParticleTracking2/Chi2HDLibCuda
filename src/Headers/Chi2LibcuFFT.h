/*
 * Chi2LibCuFFT.h
 *
 *  Created on: 09/12/2011
 *      Author: juanin
 */

#include "Container/cuMyMatrix.h"
#include "cufft.h"

#ifndef CHI2LIBCUFFT_H_
#define CHI2LIBCUFFT_H_

class Chi2LibcuFFT {
private:
	/**
	 * Maneja los errores de CUFFT.
	 * De existir algun error, este se imprime en pantalla y se cierra el programa con codigo -1.
	 * @param res resultado de la ejecucion de cuFFT.
	 */
	static void manageErrorFFT(cufftResult res);

public:
	/**
	 * Genera una convolucion en 2 dimensiones usando CUFFT.
	 * La matriz de salida generalmente debe tener como tamaño: tamaño de la imagen + tamaño de la particula ideal -1.
	 * @param img Matriz representando una imagen.
	 * @param kernel_img Matriz representando a una particula ideal.
	 * @param output Resultado de la convolucion de salida.
	 */
	static void conv2D(cuMyMatrix* img, cuMyMatrix* kernel_img, cuMyMatrix* output);

	/**
	 * Calcula la matriz de salida para obtener una imagen chi2 basandose en 3 convoluciones necesarias.
	 * @param first_term El primer termino es la convolucion entre la matriz de imagen original y la matriz de la particula ideal elevado al cuadrado.
	 * @param second_term El segundo termino corresponde a la convolucion entre la matriz de imagen original elevada al cuadrado y la matriz de la particula ideal.
	 * @param third_term El tercer termino y que se debe calcular una sola vez, corresponde a la convolucion entre una matriz de ceros y la amtriz de la particula ideal elevada al cubo.
	 * @param output Matriz de salida.
	 */
	static void fftresutl(cuMyMatrix* first_term, cuMyMatrix* second_term, cuMyMatrix* third_term, cuMyMatrix* output);
};

#endif /* CHI2LIBCUFFT_H_ */
