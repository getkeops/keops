This folder contains a new implementation of the Cuda libkp library which allows to write more general kernel convolutions. 
It is still under development and not currently compatible with the previous version. Only a test executable is provided for the moment. 

The stand-alone programm test.cu performs a single convolution in the GPU. To compile it, use the command
nvcc -std=c++11 -o test test.cu
Different types of kernels and dimensions can be used. They are set via #define macros at the beginning of the file.





	
	



