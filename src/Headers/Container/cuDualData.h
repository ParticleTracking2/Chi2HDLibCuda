/*
 * cuDualData.h
 *
 *  Created on: 12/12/2011
 *      Author: juanin
 */
#include "../Chi2LibcuUtils.h"

#ifndef CUDUALDATA_H_
#define CUDUALDATA_H_

struct DualDatai{
	int* d_data;
	int* h_data;
};

struct DualDataf{
	float* d_data;
	float* h_data;
};

DualDatai DualData_CreateInt();

DualDataf DualData_CreateFloat();

void DualData_CopyToHost(DualDatai data);

void DualData_CopyToDevice(DualDatai data);

void DualData_CopyToHost(DualDataf data);

void DualData_CopyToDevice(DualDataf data);

void DualData_Destroy(DualDatai data);

void DualData_Destroy(DualDataf data);


#endif /* CUDUALDATA_H_ */
