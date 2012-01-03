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

/**
 * Estructura de datos para almacenar informacion del dispositivo.
 */
struct DeviceProps{
	/**
	 * Nombre del dispositivo.
	 */
	char name[256];

	/**
	 * Identificador del dispositivo.
	 */
	int device;
};

/**
 * Clase con funciones especificas para ejecutar el algoritmo de minimos cuadrados simple.
 * Todos los elementos se procesarán en GPU, o la mayoria de ellos.
 */
class Chi2Libcu {
public:
	/**
	 * Establece el dispositivo que se ocupara en todas las futuras llamadas.
	 * @param device Identificador del dispositivo a utilizar [0, 1, ...]
	 */
	static void setDevice(int device);

	/**
	 * Retorna el nombre del dispositivo seleccionado.
	 * @return Estructura de datos con el nombre y el identificador del dispositivo.
	 */
	static DeviceProps getProps();

	/**
	 * Obtiene los valores maximos y minimos de una matriz de datos en GPU.
	 * @param arr Matriz de datos donde buscar.
	 * @return Par de numeros flotantes minimo y maximo en ese orden
	 */
	static pair<float, float> minMax(cuMyMatrix *arr);

	/**
	 * Normaliza una matriz de datos en GPU en base a sus valores maximos y minimos.
	 * @param arr Matriz de datos a normalizar.
	 * @param _min valor minimo de la matriz.
	 * @param _max valor maximo de la matriz.
	 */
	static void normalize(cuMyMatrix *arr, float _min, float _max);

	/**
	 * Genera una matriz representante de una particula ideal en GPU.
	 * @param ss Tamaño de la particula ideal.
	 * @param os Tamaño /2 de la particula
	 * @param d Diametro de la particula ideal.
	 * @param w Peso de la particula, este valor depende del foco de la imagen.
	 * @return Matriz en GPU representando una particula ideal.
	 */
	static cuMyMatrix gen_kernel(unsigned int ss, unsigned int os, float d, float w);

	/**
	 * Obentiene los minimos locales validos dentro una imagen chi2, detectando asi, los peaks.
	 * @see validatePeaks()
	 * @param arr Matriz de imagen en donde detectar los peaks.
	 * @param threshold Tolerancia minima aceptable de intensidad de imagen para detectar un peak.
	 * @param mindistance Area minima aceptable para considerar un peak valido como minimo local.(Deprecado)
	 * @param minsep Separacion Minima aceptable entre un peak y otro para ser considerado valido.
	 * @return Contenedor de cuMyPeaks detectados en GPU.
	 */
	static cuMyPeakArray getPeaks(cuMyMatrix *arr, int threshold, int mindistance, int minsep);

	/**
	 * Valida que los peaks tengan un minimo de separacion.
	 * @param peaks Contenedor de Peaks a validar.
	 * @param mindistance Area minima aceptable para considerar un peak valido como minimo local.
	 */
	static void validatePeaks(cuMyPeakArray* peaks, int mindistance);

	/**
	 * Genera las matrices auxiliares con valores iguales a las distancias en X e Y al centro de los peaks detectados y el indice del peak.
	 * Actualmente la implementacion no es 100% confiable debido a errores de sincronizacion que no dejan bien los datos.
	 * @param peaks Contenedor de Peaks detectados en GPU.
	 * @param shift correccion de posicion para los peaks dentro de la imagen original. (Peaks tienen posiciones en la imagen Chi2).
	 * @param grid_x Matriz en GPU donde guardar las posiciones de X al centro del peak mas cercano.
	 * @param grid_y Matriz en GPU donde guardar las posiciones de Y al centro del peak mas cercano.
	 * @param over Matriz en GPU donde guardar los indices dentro del vector de peaks de los peaks mas cercanos.
	 */
	static void generateGrid(cuMyPeakArray* peaks, unsigned int shift, cuMyMatrix* grid_x, cuMyMatrix* grid_y, cuMyMatrixi* over);

	/**
	 * Calcula la diferencia de la Imagen Chi2 generada a partir de los peaks detectados y la Imagen normal.
	 * @param img Imagen original de la deteccion de peaks.
	 * @param grid_x Posiciones de X al centro del peak mas cercano.
	 * @param grid_y Posiciones de Y al centro del peak mas cercano.
	 * @param d Diametro de la particula ideal.
	 * @param w Peso de la particula, este valor depende del foco de la imagen.
	 * @param diffout Matriz donde guardar la diferencia.
	 * @return valor del error cuadrado.
	 */
	static float computeDifference(cuMyMatrix *img,cuMyMatrix *grid_x, cuMyMatrix *grid_y, float d, float w, cuMyMatrix *diffout);

	/**
	 * Trata de mejorar el centro de las particulas mediante el metodo de newton.
	 * @param over Matriz de indices dentro del vector de peaks de los peaks mas cercanos.
	 * @param diff Matriz de diferencia entre Imagen original y generada a partir de los Peaks.
	 * @param peaks Peaks detectados actualmente en GPU.
	 * @param shift correccion de posicion para los peaks dentro de la imagen original. (Peaks tienen posiciones en la imagen Chi2).
	 * @param d Diametro de la particula ideal.
	 * @param w Peso de la particula, este valor depende del foco de la imagen.
	 * @param dp
	 * @param maxdr Valor Maximo para considerar la nueva posicion admisible.
	 */
	static void newtonCenter(cuMyMatrixi *over, cuMyMatrix *diff, cuMyPeakArray *peaks, int shift, float D, float w, float dp, float maxdr = 20.0);

};

#endif /* CHI2LIBCU_H_ */
