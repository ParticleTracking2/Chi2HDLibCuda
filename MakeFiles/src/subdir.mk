################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Chi2Libcu.cu \
../src/Chi2LibcuFFT.cu \
../src/Chi2LibcuHighDensity.cu \
../src/Chi2LibcuMatrix.cu \
../src/Chi2LibcuUtils.cu 

OBJS += \
./src/Chi2Libcu.o \
./src/Chi2LibcuFFT.o \
./src/Chi2LibcuHighDensity.o \
./src/Chi2LibcuMatrix.o \
./src/Chi2LibcuUtils.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVIDIA CUDA Compiler'
	nvcc -I/usr/local/cuda/include -I/usr/include/c++/4.4 -I/usr/include -I/usr/local/include -O3 -c -Xcompiler -fmessage-length=0 -arch compute_13 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


