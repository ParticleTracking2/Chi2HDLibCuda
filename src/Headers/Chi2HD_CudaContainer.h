/*
 * Chi2HD_CudaContainer.h
 *
 *  Created on: 07/12/2011
 *      Author: juanin
 */

#ifndef CHI2HD_CUDA_CONTAINER_
#define CHI2HD_CUDA_CONTAINER_
#include <stdlib.h>

using namespace std;

class cuMyMatrix{
private:
	float * _device_array;
	float * _host_array;
	int device;
	unsigned int _sizeX;
	unsigned int _sizeY;
	void allocateDevice();
	void allocateHost();
	void goEmpty();
public:
	cuMyMatrix();
	cuMyMatrix(unsigned int x, unsigned int y);
	cuMyMatrix(unsigned int x, unsigned int y, float def);
	~cuMyMatrix();

	void allocate(unsigned int x, unsigned int y);
	void allocateDevice(unsigned int x, unsigned int y);
	void allocateHost(unsigned int x, unsigned int y);

	void deallocate();
	void deallocateHost();
	void deallocateDevice();

	unsigned int sizeX();
	unsigned int sizeY();

	void copyToDevice();
	void copyToHost();

	void reset(float def = 0);

	float getValueDevice(unsigned int index);
	float getValueDevice(unsigned int x, unsigned int y);
	float & atDevice(unsigned int x, unsigned int y);

	float getValueHost(unsigned int index);
	float getValueHost(unsigned int x, unsigned int y);
	float & atHost(unsigned int x, unsigned int y);
};

#endif
