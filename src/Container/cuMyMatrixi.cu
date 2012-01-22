
#include "../Headers/Container/cuMyMatrix.h"
#include "../Headers/Chi2LibcuUtils.h"


void cuMyMatrixi::goEmpty(){
	_device_array = 0;
	_host_array = 0;
	_sizeX = 0; _sizeY = 0;
	device = -1;
}
/**
 *******************************
 * Constructores y Destructores
 *******************************
 */
cuMyMatrixi::cuMyMatrixi(){
	goEmpty();
}

cuMyMatrixi::cuMyMatrixi(unsigned int x, unsigned int y){
	goEmpty();
	_sizeX = x; _sizeY = y;
	allocateDevice();
}

cuMyMatrixi::cuMyMatrixi(unsigned int x, unsigned int y, int def){
	goEmpty();
	_sizeX = x; _sizeY = y;
	allocateDevice();
	reset(def);
}

void cuMyMatrixi::allocateDevice(){
	if(_sizeX*_sizeY > 0){
		cudaError_t err = cudaMalloc((void**)&_device_array, (size_t)(_sizeY*_sizeX*sizeof(int)));
		manageError(err);
	}
}

void cuMyMatrixi::allocateHost(){
	if(_sizeX*_sizeY > 0){
		_host_array = (int*)malloc(_sizeY*_sizeX*sizeof(int));
	}
}

void cuMyMatrixi::allocate(unsigned int x, unsigned int y){
	allocateDevice(x,y);
	allocateHost(x,y);
}

void cuMyMatrixi::allocateDevice(unsigned int x, unsigned int y){
	_sizeX = x; _sizeY = y;
	allocateDevice();
}

void cuMyMatrixi::allocateHost(unsigned int x, unsigned int y){
	_sizeX = x; _sizeY = y;
	allocateHost();
}

cuMyMatrixi::~cuMyMatrixi(){
	deallocate();
}

void cuMyMatrixi::deallocate(){
	deallocateDevice();
	deallocateHost();
}

void cuMyMatrixi::deallocateDevice(){
	if(_device_array){
		cudaError_t err = cudaFree(_device_array);
		manageError(err);
	}
	_device_array = 0;
	if(!_host_array){
		_sizeX = 0; _sizeY = 0;
	}
}

void cuMyMatrixi::deallocateHost(){
	if(_host_array)
		free(_host_array);
	_host_array = 0;
	if(!_device_array){
		_sizeX = 0; _sizeY = 0;
	}
}

void cuMyMatrixi::operator = (cuMyMatrixi mtrx){
	allocateDevice(mtrx.sizeX(), mtrx.sizeY());
	cudaError_t err = cudaMemcpy(_device_array, mtrx.devicePointer(), _sizeY*_sizeX*sizeof(int), cudaMemcpyDeviceToDevice);
	manageError(err);
}

/**
 *******************************
 * Metodos
 *******************************
 */
unsigned int cuMyMatrixi::size(){
	return _sizeX*_sizeY;
}

unsigned int cuMyMatrixi::sizeX(){
	return _sizeX;
}

unsigned int cuMyMatrixi::sizeY(){
	return _sizeY;
}

void cuMyMatrixi::copyToDevice(){
	if(!_device_array){
		allocateDevice();
	}
	cudaError_t err = cudaMemcpy(_device_array, _host_array, _sizeY*_sizeX*sizeof(int), cudaMemcpyHostToDevice);
	manageError(err);
}

void cuMyMatrixi::copyToHost(){
	if(!_host_array){
		allocateHost();
	}
	cudaError_t err = cudaMemcpy(_host_array, _device_array, _sizeY*_sizeX*sizeof(int), cudaMemcpyDeviceToHost);
	manageError(err);
}

__global__ void __cuMyMatrixi_reset(int* arr, int size, int def){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		arr[idx] = def;
}

void cuMyMatrixi::reset(int def){
	dim3 dimGrid(_findOptimalGridSize(_sizeY*_sizeX));
	dim3 dimBlock(_findOptimalBlockSize(_sizeY*_sizeX));
	__cuMyMatrixi_reset<<<dimGrid, dimBlock>>>(_device_array, _sizeY*_sizeX, def);
	checkAndSync();
}

int* cuMyMatrixi::devicePointer(){
	return _device_array;
}

int* cuMyMatrixi::hostPointer(){
	return _host_array;
}

/**
 * Valores en Host
 */
int cuMyMatrixi::getValueHost(unsigned int index){
	return _host_array[index];
}

int cuMyMatrixi::getValueHost(unsigned int x, unsigned int y){
	return _host_array[x+y*_sizeY];
}

int & cuMyMatrixi::atHost(unsigned int x, unsigned int y){
	return _host_array[x+y*_sizeY];
}

int & cuMyMatrixi::atHost(unsigned int index){
	return _host_array[index];
}
