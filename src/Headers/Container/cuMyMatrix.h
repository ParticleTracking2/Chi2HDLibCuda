/*
 * cuMyMatrix.h
 *
 *  Created on: 07/12/2011
 *      Author: juanin
 */

#ifndef CUMYMATRIX_
#define CUMYMATRIX_
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
	cuMyMatrix(float* arr, unsigned int sizex, unsigned int sizey);
	cuMyMatrix(unsigned int x, unsigned int y);
	cuMyMatrix(unsigned int x, unsigned int y, float def);
	~cuMyMatrix();

	void allocate(unsigned int x, unsigned int y);
	void allocateDevice(unsigned int x, unsigned int y);
	void allocateHost(unsigned int x, unsigned int y);

	void deallocate();
	void deallocateHost();
	void deallocateDevice();

	unsigned int size();
	unsigned int sizeX();
	unsigned int sizeY();

	void copyToDevice();
	void copyToHost();

	void reset(float def = 0);
	float* devicePointer();
	float* hostPointer();

	float getValueHost(unsigned int index);
	float getValueHost(unsigned int x, unsigned int y);
	float & atHost(unsigned int x, unsigned int y);
	float & atHost(unsigned int index);
};

class cuMyMatrixi{
private:
	int * _device_array;
	int * _host_array;
	int device;
	unsigned int _sizeX;
	unsigned int _sizeY;
	void allocateDevice();
	void allocateHost();
	void goEmpty();
public:
	cuMyMatrixi();
	cuMyMatrixi(int* arr, unsigned int sizex, unsigned int sizey);
	cuMyMatrixi(unsigned int x, unsigned int y);
	cuMyMatrixi(unsigned int x, unsigned int y, int def);
	~cuMyMatrixi();

	void allocate(unsigned int x, unsigned int y);
	void allocateDevice(unsigned int x, unsigned int y);
	void allocateHost(unsigned int x, unsigned int y);

	void deallocate();
	void deallocateHost();
	void deallocateDevice();

	unsigned int size();
	unsigned int sizeX();
	unsigned int sizeY();

	void copyToDevice();
	void copyToHost();

	void reset(int def = 0);
	int* devicePointer();
	int* hostPointer();

	int getValueHost(unsigned int index);
	int getValueHost(unsigned int x, unsigned int y);
	int & atHost(unsigned int x, unsigned int y);
	int & atHost(unsigned int index);
};

#endif
