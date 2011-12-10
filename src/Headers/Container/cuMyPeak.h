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

struct cuMyPeakArray{
	cuMyPeak* _host_array;
	cuMyPeak* _device_array;
	unsigned int size;
};

#endif /* CUMYPEAK_H_ */
