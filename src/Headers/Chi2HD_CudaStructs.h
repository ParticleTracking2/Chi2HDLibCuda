/*
 * Chi2HD_CudaStructs.h
 *
 *  Created on: 23/11/2011
 *      Author: juanin
 */

#ifndef CHI2HD_CUDA_STRUCTS_
#define CHI2HD_CUDA_STRUCTS_
#include <stdlib.h>

/**
 * Mini Array en cuda
 */
struct cuMyArray2D{
	unsigned int _sizeX;
	unsigned int _sizeY;

	float * _device_array;
	float * _host_array;
	int _device;

	cuMyArray2D(){
		_device_array = 0;
		_host_array = 0;
		_device = 0;
		_sizeX = 0;
		_sizeY = 0;
	}

	unsigned int getSize(){
		return _sizeX*_sizeY;
	}

	// Cuidado con esta operacion, si tiene mal los parametros...
	unsigned int position(unsigned int x, unsigned int y){
		return x+_sizeY*y;
	}

	unsigned int safePosition(unsigned int x, unsigned int y){
		unsigned int _x = x%_sizeX;
		unsigned int _y = y%_sizeY;
		if(_x*_y <= _sizeX*_sizeY)
			return _x+_sizeY*_y;
		else
			return 0;
	}

	float getValueHost(unsigned int x, unsigned int y){
		unsigned int pos = position(x,y);
		return _host_array[pos];
	}

	float getValueHost(unsigned int position){
		return _host_array[position];
	}

	void freeHost(){
		free(_host_array);
		_host_array = 0;
	}
};

struct myPair{
	float first;
	float second;
};

struct cuPeak{
	int x,y;
	float fx, fy;
	float dfx, dfy;

	float chi_intensity;
	float img_intensity;

	float vor_area;
	bool solid;
};

#endif
