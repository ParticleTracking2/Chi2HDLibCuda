#! /bin/sh

InstallDependingLibraries()
{

	echo "========================"
	echo "Installing libraries (tested on Ubuntu 16.04)"
	sudo apt-get install gcc g++ imagemagick libmagick++-dev libmagickwand-dev libfftw3-dev liblog4cpp5v5 liblog4cpp5-dev libxmu-dev libxi-dev freeglut3 freeglut3-dev libqhull-dev
}

InstallCUDA80()
{
	mkdir CUDA_Tools
	cd CUDA_Tools

	echo "========================"
	echo " Downloading CUDA Toolkit 8.0"
	wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
	
	echo "========================"
	echo " Running Installer."
	echo " Please follow the instructions. Leave the Paths by default and answer Yes to all"
	sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
	sudo apt-get update
	sudo apt-get install cuda

	echo "========================"
	echo " Adding CUDA Binary PATH"
	echo "PATH=\"$PATH:/usr/local/cuda-8.0/bin\"" | sudo tee -a /etc/environment
	source /etc/environment

	echo "========================"
	echo " Adding CUDA Library PATH"
	sudo touch /etc/ld.so.conf.d/nvidia_cuda80.conf
	echo "/usr/local/cuda-8.0/lib64" | sudo tee -a /etc/ld.so.conf.d/nvidia_cuda80.conf
	echo "/usr/local/cuda-8.0/lib" | sudo tee -a /etc/ld.so.conf.d/nvidia_cuda80.conf
	sudo ldconfig

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
	echo "Usage: ./install.sh"
}

Main()
{

	echo "========================"
	echo " Starting to Install CUDA Version 8.0"

	InstallDependingLibraries
	InstallCUDA80
	InstallCHI2HDCudaLib
	
}

Main
