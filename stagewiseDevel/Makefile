compile:
	nvcc -ccbin g++ -g -G -dc -m64 -o massif_full_cube_callbacks_sampling_doubleprec.o -c massif_full_cube_callbacks_sampling_doubleprec.cu
	nvcc -ccbin g++ -g -G -m64 -o massif.x massif_full_cube_callbacks_sampling_doubleprec.o -lcufft_static -lculibos -lfftw3
