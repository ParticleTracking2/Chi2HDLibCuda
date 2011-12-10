/*
 * Chi2LibcuMatrix.h
 *
 *  Created on: 09/12/2011
 *      Author: juanin
 */

#include "Container/cuMyMatrix.h"

#ifndef CHI2LIBCUMATRIX_H_
#define CHI2LIBCUMATRIX_H_

class Chi2LibcuMatrix {
public:
	/**
	 * Eleva al cuadrado la matriz
	 */
	static void squareIt(cuMyMatrix *mtrx);
	/**
	 * Eleva al cubo la matriz
	 */
	static void cubeIt(cuMyMatrix *mtrx);
	/**
	 * Copia una matriz a la otra matriz
	 */
	static void copy(cuMyMatrix *in, cuMyMatrix *out);
	static void copy(cuMyMatrixi *in, cuMyMatrixi *out);
};


#endif /* CHI2LIBCUMATRIX_H_ */
