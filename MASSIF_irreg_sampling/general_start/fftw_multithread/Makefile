
compile:
	nvcc -ccbin icc -std=c++11 -g -G -dc -m64 -o main_massif.o -c main_massif.cu
	nvcc -ccbin icc -std=c++11 -g -G -m64 -o massif.x main_massif.o -lcufft_static -lculibos -lfftw3_threads -lfftw3 -lpthread -lm 
clean:
	rm -rf massif *.o
