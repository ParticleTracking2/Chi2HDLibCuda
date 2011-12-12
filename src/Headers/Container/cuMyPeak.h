/*
 * cuMyPeak.h
 *
 *  Created on: 10/12/2011
 *      Author: juanin
 */

#ifndef CUMYPEAK_H_
#define CUMYPEAK_H_

struct cuMyPeak{
	unsigned int lineal_index;
	int x,y;
	float fx, fy;
	float dfx, dfy;

	float chi_intensity;
	float img_intensity;

	float vor_area;
	bool solid;
	bool valid;
};

class cuMyPeakArray{
private:
	cuMyPeak* _host_array;
	cuMyPeak* _device_array;
	unsigned int _size;
	void allocateDevice();
	void allocateHost();
	void goEmpty();
public:
	cuMyPeakArray();
	cuMyPeakArray(unsigned int size);
	~cuMyPeakArray();

	void copyToHost();
	void copyToDevice();
	void deallocateDevice();
	void deallocateHost();

	unsigned int size();
	cuMyPeak* devicePointer();
	cuMyPeak* hostPointer();

	cuMyPeak getHostValue(unsigned int index);
	cuMyPeak & atHost(unsigned int index);
};

#endif /* CUMYPEAK_H_ */
