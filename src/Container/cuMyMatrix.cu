
#include "../Headers/Container/cuMyMatrix.h"
#include "../Headers/Chi2LibcuUtils.h"


void cuMyMatrix::goEmpty(){
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
cuMyMatrix::cuMyMatrix(){
	goEmpty();
}

cuMyMatrix::cuMyMatrix(float* arr, unsigned int sizex, unsigned int sizey){
	goEmpty();
	_sizeX = sizex; _sizeY = sizey;
	allocateDevice();
	cudaError_t err = cudaMemcpy(_device_array, arr, _sizeY*_sizeX*sizeof(float), cudaMemcpyHostToDevice);
	manageError(err);
}

cuMyMatrix::cuMyMatrix(unsigned int x, unsigned int y){
	goEmpty();
	_sizeX = x; _sizeY = y;
	allocateDevice();
}

cuMyMatrix::cuMyMatrix(unsigned int x, unsigned int y, float def){
	goEmpty();
	_sizeX = x; _sizeY = y;
	allocateDevice();
	reset(def);
}

void cuMyMatrix::allocateDevice(){
	if(_sizeX*_sizeY > 0){
		cudaError_t err = cudaMalloc((void**)&_device_array, (size_t)(_sizeY*_sizeX*sizeof(float)));
		manageError(err);
	}
}

void cuMyMatrix::allocateHost(){
	if(_sizeX*_sizeY > 0){
		_host_array = (float*)malloc(_sizeY*_sizeX*sizeof(float));
	}
}

void cuMyMatrix::allocate(unsigned int x, unsigned int y){
	allocateDevice(x,y);
	allocateHost(x,y);
}

void cuMyMatrix::allocateDevice(unsigned int x, unsigned int y){
	_sizeX = x; _sizeY = y;
	allocateDevice();
}

void cuMyMatrix::allocateHost(unsigned int x, unsigned int y){
	_sizeX = x; _sizeY = y;
	allocateHost();
}

cuMyMatrix::~cuMyMatrix(){
	deallocate();
}

void cuMyMatrix::deallocate(){
	deallocateDevice();
	deallocateHost();
}

void cuMyMatrix::deallocateDevice(){
	if(_device_array){
		cudaError_t err = cudaFree(_device_array);
		manageError(err);
	}
	_device_array = 0;
}

void cuMyMatrix::deallocateHost(){
	if(_host_array)
		free(_host_array);
	_host_array = 0;
}

void cuMyMatrix::operator = (cuMyMatrix mtrx){
	allocateDevice(mtrx.sizeX(), mtrx.sizeY());
	cudaError_t err = cudaMemcpy(_device_array, mtrx.devicePointer(), _sizeY*_sizeX*sizeof(float), cudaMemcpyDeviceToDevice);
	manageError(err);
}

/**
 *******************************
 * Metodos
 *******************************
 */
unsigned int cuMyMatrix::size(){
	return _sizeX*_sizeY;
}

unsigned int cuMyMatrix::sizeX(){
	return _sizeX;
}

unsigned int cuMyMatrix::sizeY(){
	return _sizeY;
}

void cuMyMatrix::copyToDevice(){
	if(!_device_array){
		allocateDevice();
	}
	cudaError_t err = cudaMemcpy(_device_array, _host_array, _sizeY*_sizeX*sizeof(float), cudaMemcpyHostToDevice);
	manageError(err);
}

void cuMyMatrix::copyToHost(){
	if(!_host_array){
		allocateHost();
	}
	cudaError_t err = cudaMemcpy(_host_array, _device_array, _sizeY*_sizeX*sizeof(float), cudaMemcpyDeviceToHost);
	manageError(err);
}

__global__ void __cuMyMatrix_reset(float* arr, int size, float def){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		arr[idx] = def;
}

void cuMyMatrix::reset(float def){
	dim3 dimGrid(_findOptimalGridSize(_sizeY*_sizeX));
	dim3 dimBlock(_findOptimalBlockSize(_sizeY*_sizeX));
	__cuMyMatrix_reset<<<dimGrid, dimBlock>>>(_device_array, _sizeY*_sizeX, def);
	checkAndSync();
}

float* cuMyMatrix::devicePointer(){
	return _device_array;
}

float* cuMyMatrix::hostPointer(){
	return _host_array;
}

/**
 * Valores en Host
 */
float cuMyMatrix::getValueHost(unsigned int index){
	return _host_array[index];
}

float cuMyMatrix::getValueHost(unsigned int x, unsigned int y){
	return _host_array[x+y*_sizeY];
}

float & cuMyMatrix::atHost(unsigned int x, unsigned int y){
	return _host_array[x+y*_sizeY];
}

float & cuMyMatrix::atHost(unsigned int index){
	return _host_array[index];
}
