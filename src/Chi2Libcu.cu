/*
 * Chi2Libcu.cu
 *
 *  Created on: 10/12/2011
 *      Author: juanin
 */
#include "Headers/Container/cuDualData.h"
#include "Headers/Chi2Libcu.h"
#include "Headers/Chi2LibcuUtils.h"

/******************
 * Min Max
 ******************/
pair<float, float> Chi2Libcu::minMax(cuMyMatrix *arr){
	pair<float, float> ret;
	arr->copyToHost();

	float tempMax = arr->getValueHost(0);
	float tempMin = arr->getValueHost(0);
	for(unsigned int i=0; i < arr->size(); ++i){
		float tmp = arr->getValueHost(i);
		if(tempMax < tmp)
			tempMax = tmp;
		if(tempMin > tmp)
			tempMin = tmp;
	}
	ret.first = tempMin;
	ret.second = tempMax;

	return ret;
}
/******************
 * Normalizar
 ******************/
__global__ void __normalize(float* arr, unsigned int size, float _min, float _max){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float dif = _max - _min;
	if(idx < size)
		arr[idx] = (float)((1.0f*_max - arr[idx]*1.0f)/dif);
}

void Chi2Libcu::normalize(cuMyMatrix *arr, float _min, float _max){
	dim3 dimGrid(_findOptimalGridSize(arr->size()));
	dim3 dimBlock(_findOptimalBlockSize(arr->size()));
	__normalize<<<dimGrid, dimBlock>>>(arr->devicePointer(), arr->size(), _min, _max);
	cudaError_t err = cudaGetLastError(); manageError(err);
	err = cudaDeviceSynchronize();	manageError(err);
}


/******************
 * Kernel
 ******************/
__global__ void __gen_kernel(float* arr, unsigned int size, unsigned int ss, unsigned int os, float d, float w){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		float absolute = abs(sqrtf( (idx%ss-os)*(idx%ss-os) + (idx/ss-os)*(idx/ss-os) ));
		arr[idx] = (1.0f - tanhf((absolute - d/2.0f)/w))/2.0f;
	}
}

cuMyMatrix Chi2Libcu::gen_kernel(unsigned int ss, unsigned int os, float d, float w){
	cuMyMatrix kernel(ss,ss);
	dim3 dimGrid(1);
	dim3 dimBlock(ss*ss);
	__gen_kernel<<<dimGrid, dimBlock>>>(kernel.devicePointer(), kernel.size(), ss, os, d, w);
	cudaError_t err = cudaGetLastError(); manageError(err);
	err = cudaDeviceSynchronize();	manageError(err);

	return kernel;
}

/******************
 * Peaks
 ******************/
__device__ bool __findLocalMinimum(float* arr, unsigned int sizeX, unsigned int sizeY, unsigned int imgX, unsigned int imgY, unsigned int idx, int minsep, int* counter){
	for(int localX = minsep; localX >= -minsep; --localX){
		for(int localY = minsep; localY >= -minsep; --localY){
			if(!(localX == 0 && localY == 0)){
				int currentX = (imgX+localX);
				int currentY = (imgY+localY);

				if(currentX < 0)
					currentX = sizeX + currentX;
				if(currentY < 0)
					currentY = sizeY + currentY;

				currentX = (currentX)% sizeX;
				currentY = (currentY)% sizeY;

				if(arr[idx] <= arr[currentX+currentY*sizeY]){
					return false;
				}
			}
		}
	}
	atomicAdd(&counter[0], 1);
	return true;
}

__global__ void __findMinimums(float* arr, unsigned int sizeX, unsigned int sizeY, int threshold, int minsep, bool* out, int* counter){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int imgX = idx%sizeX;
	int imgY = (int)floorf(idx/sizeY);

	if(idx < sizeX*sizeY && arr[idx] > threshold){
		if(__findLocalMinimum(arr, sizeX, sizeY, imgX, imgY, idx, minsep, counter))
			out[idx] = true;
		else
			out[idx] = false;
	}
}

__global__ void __fillPeakArray(float* img, bool* peaks_detected, unsigned int sizeX, unsigned int sizeY, cuMyPeak* peaks, int* counter){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < sizeX*sizeY && peaks_detected[idx]){
		cuMyPeak peak;
		peak.x = (int)floorf(idx/sizeY);
		peak.y = idx%sizeX;
		peak.chi_intensity = img[idx];
		peak.fx = peak.x;
		peak.fy = peak.y;
		peak.dfx = peak.dfy = 0;
		peak.solid = false;
		peak.valid = true;
		peaks[atomicAdd(&counter[0], 1)] = peak;
	}
}

__global__ void __validatePeaks(cuMyPeak* peaks, unsigned int size, unsigned int mindistance){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int mindistance2 = mindistance*mindistance;

	if(idx < size)
	for(unsigned int j=0; j < size && j != idx ; ++j){
		int difx = peaks[idx].x - peaks[j].x;
		int dify = peaks[idx].y - peaks[j].y;

		if( (difx*difx + dify*dify) < mindistance2){
			if(peaks[idx].chi_intensity < peaks[j].chi_intensity){
				peaks[idx].valid = false;
			}else{
				peaks[j].valid = false;
			}
			break;
		}
	}
}

struct cuMyPeakCompare {
  __host__ __device__
  bool operator()(const cuMyPeak &lhs, const cuMyPeak &rhs){
	  return lhs.chi_intensity < rhs.chi_intensity;
  }
};

cuMyPeakArray Chi2Libcu::getPeaks(cuMyMatrix *arr, int threshold, int mindistance, int minsep){
	bool* d_minimums;
	size_t arrSize = arr->size()*sizeof(bool);
	cudaError_t err = cudaMalloc((void**)&d_minimums, arrSize);
	manageError(err);
	cudaMemset(d_minimums, 0, arr->size()*sizeof(bool));

	int* h_counter; h_counter = (int*)malloc(sizeof(int));
	int* d_counter; cudaMalloc((void**)&d_counter, sizeof(int));
	cudaMemset(d_counter, 0, sizeof(int));

	// Encontrar Minimos
	dim3 dimGrid(_findOptimalGridSize(arr->size()));
	dim3 dimBlock(_findOptimalBlockSize(arr->size()));
	__findMinimums<<<dimGrid, dimBlock>>>(arr->devicePointer(), arr->sizeX(), arr->sizeY(), threshold, minsep, d_minimums, d_counter);
	err = cudaGetLastError(); manageError(err);
	err = cudaDeviceSynchronize();	manageError(err);

	// Contador de datos
	err = cudaMemcpy(h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
	manageError(err);

	// Alocar datos
	cuMyPeakArray peaks(h_counter[0]);
	cudaMemset(d_counter, 0, sizeof(int));

	__fillPeakArray<<<dimGrid, dimBlock>>>(arr->devicePointer(), d_minimums, arr->sizeX(), arr->sizeY(), peaks.devicePointer(), d_counter);
	err = cudaGetLastError(); manageError(err);
	err = cudaDeviceSynchronize();	manageError(err);

	// Ordenar de menor a mayor en intensidad de imagen Chi
	peaks.sortByChiIntensity();

	// Validar
	dim3 dimGrid2(_findOptimalGridSize(peaks.size()));
	dim3 dimBlock2(_findOptimalBlockSize(peaks.size()));
	__validatePeaks<<<dimGrid2, dimBlock2>>>(peaks.devicePointer(), peaks.size(), mindistance);
	err = cudaGetLastError(); manageError(err);
	err = cudaDeviceSynchronize();	manageError(err);

	peaks.keepValids();
	peaks.sortByChiIntensity();

	cudaFree(d_minimums); cudaFree(d_counter);
	free(h_counter);
	return peaks;
}

/******************
 * Matrices Auxiliares
 ******************/

__global__ void __generateGrid(cuMyPeak* peaks, unsigned int peaks_size, unsigned int shift, float* grid_x, float* grid_y, int* over, unsigned int sizeX, unsigned int sizeY){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= peaks_size)
		return;

	unsigned int half=(shift+2);
	int currentX, currentY;
	float currentDistance = 0.0;
	float currentDistanceAux = 0.0;

	if(peaks_size != 0){
		for(unsigned int localX=0; localX < 2*half+1; ++localX){
			for(unsigned int localY=0; localY < 2*half+1; ++localY){
				cuMyPeak currentPeak = peaks[idx];
				currentX = (int)round(currentPeak.fx) - shift + (localX - half);
				currentY = (int)round(currentPeak.fy) - shift + (localY - half);

				if( 0 <= currentX && currentX < sizeX && 0 <= currentY && currentY < sizeY ){
					int index = currentX+sizeY*currentY;
					currentDistance =
							sqrtf(grid_x[index]*grid_x[index] + grid_y[index]*grid_y[index]);

					currentDistanceAux =
							sqrtf(1.0f*(1.0f*localX-half+currentPeak.x - currentPeak.fx)*(1.0f*localX-half+currentPeak.x - currentPeak.fx) +
								  1.0f*(1.0f*localY-half+currentPeak.y - currentPeak.fy)*(1.0f*localY-half+currentPeak.y - currentPeak.fy));

					if(currentDistance >= currentDistanceAux){
						over[index] = idx+1;
						grid_x[index] = (1.0f*localX-half+currentPeak.x)-currentPeak.fx;
						grid_y[index] = (1.0f*localY-half+currentPeak.y)-currentPeak.fy;
					}
				}
			}
		}
	}
}

void Chi2Libcu::generateGrid(cuMyPeakArray* peaks, unsigned int shift, cuMyMatrix* grid_x, cuMyMatrix* grid_y, cuMyMatrixi* over){
	unsigned int maxDimension = grid_x->sizeX() > grid_x->sizeY() ? grid_x->sizeX() : grid_x->sizeY();
	grid_x->reset(maxDimension);
	grid_y->reset(maxDimension);
	over->reset(0);

	// TODO Evitar Race Conditions, generan resultados distinos en lanzamientos con iguales parametros

	dim3 dimGrid(_findOptimalGridSize(peaks->size()));
	dim3 dimBlock(_findOptimalBlockSize(peaks->size()));
	__generateGrid<<<dimGrid, dimBlock>>>(peaks->devicePointer(), peaks->size(), shift, grid_x->devicePointer(), grid_y->devicePointer(), over->devicePointer(), grid_x->sizeX(), grid_x->sizeY());
	cudaError_t err = cudaGetLastError(); manageError(err);
	err = cudaDeviceSynchronize();	manageError(err);
}

/******************
 * Chi2 Difference
 ******************/
__global__ void __computeDifference(float* img, float* grid_x, float* grid_y, float d, float w, float* diffout, unsigned int size, float* sum_reduction){
	extern __shared__ float sharedData[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;

	float temp = 0;
	if(idx < size){
		float x2y2 = sqrtf(1.0f*grid_x[idx]*grid_x[idx] + 1.0f*grid_y[idx]*grid_y[idx]);
		temp = ((1.0f-tanhf( (x2y2-d/2.0)/w )) - 2.0f*img[idx])/2.0f;
		diffout[idx] = temp;
	}

	// Calcular la suma cuadrada
	sharedData[tid] = temp*temp;
	__syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
        	sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

	if(tid == 0){
		sum_reduction[blockIdx.x] = sharedData[0];
	}
}

float Chi2Libcu::computeDifference(cuMyMatrix *img, cuMyMatrix *grid_x, cuMyMatrix *grid_y, float d, float w, cuMyMatrix *diffout){
	unsigned int griddim = _findOptimalGridSize(img->size());
	float* sum_reduction;
	float* host_reduction;
	cudaError_t err = cudaMalloc((void**)&sum_reduction, griddim*sizeof(float));
	manageError(err);

	dim3 dimGrid(griddim);
	dim3 dimBlock(_findOptimalBlockSize(img->size()));
	__computeDifference<<<dimGrid, dimBlock, griddim>>>(img->devicePointer(), grid_x->devicePointer(), grid_y->devicePointer(), d, w, diffout->devicePointer(), img->size(), sum_reduction);
	err = cudaGetLastError(); manageError(err);
	err = cudaDeviceSynchronize(); manageError(err);

	host_reduction = (float*)malloc(griddim*sizeof(float));
	err = cudaMemcpy(host_reduction, sum_reduction, griddim*sizeof(float), cudaMemcpyDeviceToHost);
	manageError(err);

	float total = 0;
	for(unsigned int i=0; i < griddim; ++i){
		total = total + host_reduction[i];
	}
	cudaFree(sum_reduction);
	free(host_reduction);

	return total;
}

/**
 * Newton Center
 */
__global__ void __newtonCenter(int* over, float* diff, unsigned int m_sizeX, unsigned int m_sizeY, cuMyPeak* peaks, unsigned int p_size, int half, float D, float n_w, float maxdr){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= p_size)
		return;

	double chix, chiy, chixx, chiyy, chixy;
	chix = chiy = chixx = chiyy = chixy = 0;

	cuMyPeak tmp_peak = peaks[idx];
	for(unsigned int localX=0; localX < (unsigned int)(2*half+1); ++localX){
		for(unsigned int localY=0; localY < (unsigned int)(2*half+1); ++localY){
			double xx, xx2, yy, yy2, rr, rr3;
			double dipx,dipy,dipxx,dipyy,dipxy;
			double sech2, tanh1;

			int currentX = (int)rintf(tmp_peak.fx) - (half-2) + (localX - half);
			int currentY = (int)rintf(tmp_peak.fy) - (half-2) + (localY - half);
			int index = currentX+currentY*m_sizeY;

			if( 0 <= currentX && currentX < m_sizeX && 0 <= currentY && currentY < m_sizeY && over[index] == idx +1){
				xx 		= 1.0*localX - half + tmp_peak.x - tmp_peak.fx;
				xx2 	= xx*xx;
				yy 		= 1.0*localY - half + tmp_peak.y - tmp_peak.fy;
				yy2 	= yy*yy;

				rr 		= sqrtf(xx2+yy2) + 2.2204E-16; // Por que ese numero?
				rr3 	= rr*rr*rr + 2.2204E-16;//2.2204E-16;

				sech2	= ( 1.0/(coshf( (rr-D/2.0)*n_w )) )*( 1.0/(coshf( (rr-D/2.0)*n_w )) );
				tanh1	= tanhf( (rr-D/2.0)*n_w );

				dipx 	=(-n_w)*( xx * sech2 / 2.0 / rr);
				dipy 	=(-n_w)*( yy * sech2 / 2.0 / rr);
				dipxx	=( n_w)*sech2*(2.0*n_w*xx2*rr*tanh1-yy2)/2.0 /rr3;
				dipyy	=( n_w)*sech2*(2.0*n_w*yy2*rr*tanh1-xx2)/2.0 /rr3;
				dipxy	=( n_w)*xx*yy*sech2*(2.0*n_w*rr*tanh1+1.0)/2.0/ rr3;

				float diffi = diff[index];
				chix 	+= diffi * dipx;
				chiy 	+= diffi * dipy;
				chixx	+= dipx*dipx + diffi*dipxx;
				chiyy	+= dipy*dipy + diffi*dipyy;
				chixy	+= dipx*dipy + diffi*dipxy;
			}
		}
	}

	peaks[idx].dfx = 0.0;
	peaks[idx].dfy = 0.0;
	double det = chixx*chiyy-chixy*chixy;
	if(fabsf(det) < 1E-12){
		// detproblem++;
	}else{
		peaks[idx].dfx = (chix*chiyy-chiy*chixy)/det;
		peaks[idx].dfy = (chix*(-chixy)+chiy*chixx)/det;
		float currentDPX = peaks[idx].dfx;
		float currentDPY = peaks[idx].dfy;
		float root = sqrtf(currentDPX*currentDPX + currentDPY*currentDPY);
		if(root > maxdr){
			peaks[idx].dfx /= root;
			peaks[idx].dfy /= root;
		}
	}
}

void Chi2Libcu::newtonCenter(cuMyMatrixi *over, cuMyMatrix *diff, cuMyPeakArray *peaks, int shift, float D, float w, float dp, float maxdr){
	int half = shift+2;
	float n_w = 1.0f/w;

	dim3 dimGrid(_findOptimalBlockSize(peaks->size())*2);
	dim3 dimBlock(_findOptimalBlockSize(peaks->size())/2);
	__newtonCenter<<<dimGrid, dimBlock>>>(over->devicePointer(), diff->devicePointer(), over->sizeX(), over->sizeY(), peaks->devicePointer(), peaks->size(), half, D, n_w, maxdr);
	cudaError_t err = cudaGetLastError(); manageError(err);
	err = cudaDeviceSynchronize(); manageError(err);
}
