#! /bin/sh

InstallCompilers()
{
	echo "========================"
	echo " Installing G++ Verion 4.4 and 4.8"
	sudo apt-get install g++-4.4
	sudo apt-get install g++-4.8

	echo "========================"
	echo " Installing GCC Verion 4.4 and 4.8"
	sudo apt-get install gcc-4.4
	sudo apt-get install gcc-4.8
}

SwitchTo44()
{
	echo "========================"
	echo " Switching to G++ 4.4 and GCC 4.4"
	sudo rm /usr/bin/gcc
	sudo rm /usr/bin/g++
	sudo ln -s /usr/bin/gcc-4.4 /usr/bin/gcc
	sudo ln -s /usr/bin/g++-4.4 /usr/bin/g++
}

SwitchTo48()
{
	echo "========================"
	echo " Switching to G++ 4.8 and GCC 4.8"
	sudo rm /usr/bin/gcc
	sudo rm /usr/bin/g++
	sudo ln -s /usr/bin/gcc-4.8 /usr/bin/gcc
	sudo ln -s /usr/bin/g++-4.8 /usr/bin/g++
}

InstallDependingLibraries()
{
	echo "========================"
	echo " Installing XMU"
	sudo apt-get install libxmu-dev
	echo "========================"
	echo " Installing Xi"
	sudo apt-get install libxi-dev
	echo "========================"
	echo " Installing Glut 3"
	sudo apt-get install freeglut3
	echo "========================"
	echo " Installing Glut 3 Dev"
	sudo apt-get install freeglut3-dev
}

InstallCUDA65()
{
	mkdir CUDA_Tools
	cd CUDA_Tools

	echo "========================"
	echo " Downloading CUDA Toolkit 6.5"
	wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run
	sudo chmod 777 cuda_6.5.14_linux_64.run
	
	echo "========================"
	echo " Removing All NVIDIA Drivers"
	sudo apt-get --purge remove nvidia-*
	echo "========================"
	echo " Stopping X-Server"
	sudo service lightdm stop

	echo "========================"
	echo " Running Installer."
	echo " Please follow the instructions. Leave the Paths by default and answer Yes to all"
	sudo ./cuda_6.5.14_linux_64.run -override

	echo "========================"
	echo " Adding CUDA Binary PATH"
	sudo echo "PATH=\"$PATH:/usr/local/cuda-6.5/bin\"" > /etc/environment
	source /etc/environment

	echo "========================"
	echo " Adding CUDA Library PATH"
	sudo touch /etc/ld.so.conf.d/nvidia_cuda.conf
	sudo echo "/usr/local/cuda-6.5/lib64" >> /etc/ld.so.conf.d/nvidia_cuda65.conf
	sudo echo "/usr/local/cuda-6.5/lib" >> /etc/ld.so.conf.d/nvidia_cuda65.conf
	sudo ldconfig

	mv ~/NVIDIA_CUDA-6.5_Samples CUDA_Tools/NVIDIA_CUDA-6.5_Samples
	cd NVIDIA_CUDA-6.5_Samples
	echo "========================"
	echo " Compiling Examples."
	make
	./1_Utilities/deviceQuery/deviceQuery
	cd ..

	cd ..
}

InstallCUDA40()
{
	mkdir CUDA_Tools
	cd CUDA_Tools

	echo "========================"
	echo " Downloading CUDA Toolkit 4.0"
	wget http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_64_ubuntu10.10.run
	sudo chmod 777 cudatoolkit_4.0.17_linux_64_ubuntu10.10.run
	echo "========================"
	echo " Downloading CUDA Samples 4.0"
	wget http://developer.download.nvidia.com/compute/cuda/4_0/sdk/gpucomputingsdk_4.0.17_linux.run
	sudo chmod 777 gpucomputingsdk_4.0.17_linux.run

	echo "========================"
	echo " Running Installer for CUDA Toolkit 4.0"
	echo " Please follow the instructions and leave the Paths by default"
	sudo ./cudatoolkit_4.0.17_linux_64_ubuntu10.10.run

	echo "========================"
	echo " Running Installer for CUDA Samples 4.0"
	echo " Please follow the instructions and leave the Paths by default"
	sudo ./gpucomputingsdk_4.0.17_linux.run

	echo "========================"
	echo " Adding CUDA Binary PATH"
	sudo echo "PATH=\"$PATH:/usr/local/cuda/bin\"" > /etc/environment
	source /etc/environment
	echo "========================"
	echo " Adding CUDA Library PATH"
	sudo touch /etc/ld.so.conf.d/nvidia_cuda.conf
	sudo echo "/usr/local/cuda/lib64/" >> /etc/ld.so.conf.d/nvidia_cuda40.conf
	sudo echo "/usr/local/cuda/lib/" >> /etc/ld.so.conf.d/nvidia_cuda40.conf
	sudo ldconfig

	echo "========================"
	echo " Compiling Examples"
	cd NVIDIA_GPU_Computing_SDK/
	make
	./C/bin/linux/release/deviceQuery
	cd ..
	
	cd ..
}

InstallCHI2HDCudaLib()
{
	echo "========================"
	echo " Installing Chi2HD Cuda Library"
	cd MakeFiles
	make
	sudo make install
	cd ..
}

PrintUsage()
{
	echo "Automated Install Process"
	echo "This program will install all the components necesary to compile and Run CHI2HD Cuda Library"
	echo "The platform target is Ubuntu."
	echo "Usage: ./Chi2HDCuda_install.sh [Cuda Version]"
	echo "\t Cuda Version : 40 | 65"
}

CheckArguments()
{
	if [ $# -lt 1 ]; then
		echo "Error: Not Enough Arguments"
		PrintUsage
		exit 1
	fi

	if [ "$1" != "40" ] && [ "$1" != "65" ]; then
		echo "Error: Version not match. Must be 40 or 65"
		PrintUsage
		exit 1
	fi
}

Main()
{
	CheckArguments $1

	echo "========================"
	echo " Starting to Install CUDA Version $1"

	InstallCompilers
	InstallDependingLibraries

	if [ "$1" == "40" ]; then
		SwitchTo44
		InstallCUDA40
	fi

	if [ "$1" == "65" ]; then
		SwitchTo48
		InstallCUDA65
	fi

	InstallCHI2HDCudaLib
}

Main $1