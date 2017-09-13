################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Container/cuMyMatrix.cu \
../src/Container/cuMyMatrixi.cu \
../src/Container/cuMyPeakArray.cu 

OBJS += \
./src/Container/cuMyMatrix.o \
./src/Container/cuMyMatrixi.o \
./src/Container/cuMyPeakArray.o 


# Each subdirectory must supply rules for building sources it contributes
src/Container/%.o: ../src/Container/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVIDIA CUDA Compiler'
	nvcc -I/usr/local/cuda/include -I/usr/include/c++/4.4 -I/usr/include -I/usr/local/include -O3 -c -Xcompiler -fmessage-length=0 -arch compute_30 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


