CC=g++
NVCC=nvcc
CXXFLAGS= -m64
CUDAFLAGS= -ccbin -g -G -dc
LIBS= -lcufft_static -lculibos -lfftw3


compile:
	nvcc -ccbin g++ -g -G -dc -m64 -o main_massif.o -c main_massif.cu
	nvcc -ccbin g++ -g -G -m64 -o massif.x main_massif.o -lcufft_static -lculibos -lfftw3
clean:
	rm -rf massif *.o
