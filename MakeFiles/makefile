################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include src/Container/subdir.mk
-include src/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: libChi2HD_Cuda.a

# Tool invocations
libChi2HD_Cuda.a: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC Archiver'
	ar -r  "libChi2HD_Cuda.a" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(C++_DEPS)$(OBJS)$(C_DEPS)$(CC_DEPS)$(ARCHIVES)$(CPP_DEPS)$(CXX_DEPS)$(C_UPPER_DEPS) libChi2HD_Cuda.a
	-@echo ' '
	
install:
	sudo cp libChi2HD_Cuda.a /usr/local/lib
	sudo cp -r ../src/Headers/* /usr/local/include

uninstall:
	sudo rm /usr/local/lib/libChi2HD_Cuda.a
	sudo rm /usr/local/include/Chi2LibcuFFT.h
	sudo rm /usr/local/include/Chi2Libcu.h
	sudo rm /usr/local/include/Chi2LibcuHighDensity.h
	sudo rm /usr/local/include/Chi2LibcuMatrix.h
	sudo rm /usr/local/include/Chi2LibcuUtils.h
	sudo rm -r /usr/local/include/Container

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
