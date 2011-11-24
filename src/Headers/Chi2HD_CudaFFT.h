/*
 * Chi2HD_CudaFFT.h
 *
 *  Created on: 23/11/2011
 *      Author: juanin
 */

#ifndef CHI2HD_CUDAFFT_H_
#define CHI2HD_CUDAFFT_H_

#include <stdlib.h>
#include "cufft.h"
#include "Chi2HD_CudaStructs.h"

#if defined(__cplusplus)
extern "C" {
#endif
/**
 * Maneja los errores de CUFFT
 */
void manageErrorFFT(cufftResult res);

/**
 * Genera una convolucion 2D usando CUFFT
 */
void CHI2HD_conv2D(cuMyArray2D* img, cuMyArray2D* kernel_img, cuMyArray2D* output);

/**
 * Calcula el output despues de haber calculado las 3 convoluciones necesarias.
 */
void CHI2HD_fftresutl(cuMyArray2D* first_term, cuMyArray2D* second_term, cuMyArray2D* third_term, cuMyArray2D* output);

#if defined(__cplusplus)
}
#endif

#endif
