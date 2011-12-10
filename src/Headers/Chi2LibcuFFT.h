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
	 * Maneja los errores de CUFFT
	 */
	static void manageErrorFFT(cufftResult res);

public:
	/**
	 * Genera una convolucion 2D usando CUFFT
	 */
	static void conv2D(cuMyMatrix* img, cuMyMatrix* kernel_img, cuMyMatrix* output);

	/**
	 * Calcula el output despues de haber calculado las 3 convoluciones necesarias.
	 */
	static void fftresutl(cuMyMatrix* first_term, cuMyMatrix* second_term, cuMyMatrix* third_term, cuMyMatrix* output);
};

#endif /* CHI2LIBCUFFT_H_ */
