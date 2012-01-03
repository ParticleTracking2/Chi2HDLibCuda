/*
 * cuMyPeak.h
 *
 *  Created on: 10/12/2011
 *      Author: juanin
 */
#include <vector>
#include <algorithm>

#ifndef CUMYPEAK_H_
#define CUMYPEAK_H_

/**
 * Estructura de un Peak para usar en Dispositivo.
 */
struct cuMyPeak{

	/**
	 * Posicion en X del Peak como entero.
	 */
	int x;

	/**
	 * Posicion en Y del Peak como entero.
	 */
	int y;

	/**
	 * Posicion X del Peak como float, se tiene una mayor presicion.
	 */
	float fx;

	/**
	 * Posicion Y del Peak como float, se tiene una mayor presicion.
	 */
	float fy;

	/**
	 * Delta de la posicion X del Peak, se usa para ajustar la posicion final.
	 */
	float dfx;

	/**
	 * Delta de la posicion Y del Peak, se usa para ajustar la posicion final.
	 */
	float dfy;

	/**
	 * Intensidad de imagen Chi2 del Peak.
	 */
	float chi_intensity;

	/**
	 * Area de voronoi del Peak dentro de la imagen en que se encuentra.
	 */
	float vor_area;

	/**
	 * Valor del area de Voronoi del Peak dentro de la Imagen.
	 */
	bool solid;

	/**
	 * Indicador si el Peak es valido o no.
	 * De no ser valido, este se elimina posteriormente.
	 */
	bool valid;
};

/**
 * Contenedor de Peaks para uso en Dispositivo y en HOST.
 */
class cuMyPeakArray{
private:

	/**
	 * Puntero a un arreglo de cuMyPeak en Host.
	 */
	cuMyPeak* _host_array;

	/**
	 * Puntero a un arreglo de cuMyPeak en Dispositivo.
	 */
	cuMyPeak* _device_array;

	/**
	 * Tamaño actual del conenedor.
	 */
	unsigned int _size;

	/**
	 * Reserva memoria en Dispositivo, el tamaño esta indicado por _size.
	 */
	void allocateDevice();

	/**
	 * Reserva memoria en HOST, el tamaño esta indicado por _size.
	 */
	void allocateHost();

	/**
	 * Establece todos los elementos a 0.
	 */
	void goEmpty();
public:

	/**
	 * Constructor vacio.
	 * Se debe llamar a allocate(unsigned int) para que el contenedor quede usable.
	 */
	cuMyPeakArray();

	/**
	 * Crea un contenedor de Peaks de tamaño size.
	 * @param size Tamaño del contenedor.
	 */
	cuMyPeakArray(unsigned int size);

	/**
	 * Destructor del contenedor.
	 * Libera los recursos del dispositivo y de HOST.
	 */
	~cuMyPeakArray();

	/**
	 * Agrega los datos entregados de data al final de este contenedor.
	 * El tamaño resultante es la suma del contenedor actual y el entregado.
	 * @param data Contenedor de datos a adjuntar.
	 */
	void append(cuMyPeakArray* data);

	/**
	 * Agrega todos los delta de los peaks a sus respectivos valores flotantes.
	 */
	void includeDeltas();

	/**
	 * Ordena los peaks por intensidad de imagen chi2.
	 * De menor a mayor.
	 */
	void sortByChiIntensity();

	/**
	 * Ordena los peaks por area de voronoi.
	 * De menor a mayor.
	 */
	void sortByVoronoiArea();

	/**
	 * Copia los datos del Dispositivo al Host.
	 * Si no se encuentra reservada memoria en el Host, se reserva.
	 */
	void copyToHost();

	/**
	 * Copia los datos del Host al Dispositivo.
	 * Si no se encuentra reservada memoria en el Dispositivo, se reserva.
	 */
	void copyToDevice();

	/**
	 * Reserva memoria en dispositivo del tamaño de size.
	 * @param size Tamaño a establecer del contenedor.
	 */
	void allocate(unsigned int size);

	/**
	 * Libera memoria reservada tanto del Host como del dispositivo y el contenedor queda con tamaño 0.
	 */
	void deallocate();

	/**
	 * Libera memoria reservada del dispositivo.
	 */
	void deallocateDevice();

	/**
	 * Libera memoria reservada del HOST.
	 */
	void deallocateHost();

	/**
	 * Retorna el tamaño actual del contenedor.
	 * @return Tamaño actual del contenedor.
	 */
	unsigned int size();

	/**
	 * Devuelve el puntero a los datos del contenedor del dispositivo.
	 * No se verifica la existencia de estos.
	 * @see copyToDevice()
	 * @see deallocateDevice()
	 * @return Puntero a los datos del dispositivo.
	 */
	cuMyPeak* devicePointer();

	/**
	 * Devuelve el puntero a los datos del contenedor del HOST.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @return Puntero a los datos del HOST.
	 */
	cuMyPeak* hostPointer();

	/**
	 * Limpia los peaks no validos y deja solamente los que tienen un estado valido.
	 * Posiblemente el contenedor se reduzca de tamaño.
	 */
	void keepValids();

	/**
	 * Obtiene el valor del indice del contenedor dentro del HOST.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @param index indice dentro del contenedor.
	 * @return valor del espacio dentro del contenedor indicada por index.
	 */
	cuMyPeak getHostValue(unsigned int index);

	/**
	 * Obtiene el contenido del indice del contenedor dentro del HOST.
	 * El contenido puede ser modificado.
	 * No se verifica la existencia de estos.
	 * @see copyToHost()
	 * @see deallocateHost()
	 * @param index indice dentro del contenedor.
	 * @return valor del espacio dentro del contenedor indicada por index.
	 */
	cuMyPeak & atHost(unsigned int index);
};

#endif /* CUMYPEAK_H_ */
