/*
 * Chi2LibcuHighDensity.h
 *
 *  Created on: 13/12/2011
 *      Author: juanin
 */
#include "Container/cuMyMatrix.h"

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
};


#endif /* CHI2LIBCUHIGHDENSITY_H_ */
