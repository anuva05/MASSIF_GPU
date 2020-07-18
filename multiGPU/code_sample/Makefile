compile:
	nvcc -ccbin g++ -std=c++11 -g -G -dc -m64 -o example1.o -c example1.cu
	nvcc -ccbin g++ -std=c++11 -g -G -m64 -o ex.x example1.o -lcufft_static -lculibos
clean:
	rm -rf ex *.o
