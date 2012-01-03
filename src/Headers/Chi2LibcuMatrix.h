/*
 * Chi2LibcuMatrix.h
 *
 *  Created on: 09/12/2011
 *      Author: juanin
 */

#include "Container/cuMyMatrix.h"

#ifndef CHI2LIBCUMATRIX_H_
#define CHI2LIBCUMATRIX_H_

/**
 * Clase para manejar Matrices del tipo cuMyMatrix en GPU.
 */
class Chi2LibcuMatrix {
public:

	/**
	 * Eleva al cuadrado la matriz
	 * @param mtrx Matriz a elevar al cuadrado.
	 */
	static void squareIt(cuMyMatrix *mtrx);

	/**
	 * Eleva al cubo la matriz
	 * @param mtrx Matriz a elevar al cubo.
	 */
	static void cubeIt(cuMyMatrix *mtrx);

	/**
	 * Copia una matriz a otra matriz (float).
	 * @param in Matriz origen de datos.
	 * @param out Matriz salida de datos.
	 */
	static void copy(cuMyMatrix *in, cuMyMatrix *out);

	/**
	 * Copia una matriz a otra matriz (int).
	 * @param in Matriz origen de datos.
	 * @param out Matriz salida de datos.
	 */
	static void copy(cuMyMatrixi *in, cuMyMatrixi *out);
};


#endif /* CHI2LIBCUMATRIX_H_ */
