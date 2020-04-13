
#include <iostream>
#include <string.h>
#include <math.h>
using namespace std;
// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <float.h>
#include <fftw3.h>
#define NX 512
#define NY 512
#define NZ 1024
#define K 64  //dimension of small cube
#define BATCH 1
#define NRANK 3


// h_a is array reading from exported text of matlab
cudaError_t PerfCuFFT(int argc, char **argv, cufftDoubleComplex *h_a, cufftDoubleComplex *result){
	cudaError_t cudaStatus;
	cufftHandle plan;
	cufftDoubleComplex *data;
	cufftDoubleComplex *d_a;
	int n[NRANK] = { NX, NY, NZ };
	int count;
	// Choosing CUDA device with newer architect
	//int dev = findCudaDevice(argc, (const char **)argv);


	cudaMalloc((void**)&data, sizeof(cufftDoubleComplex)*(NX*NY*NZ)*BATCH);
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		goto Error;
	}

	cudaMalloc((void**)&d_a, sizeof(cufftDoubleComplex)*NX*NY*NZ);
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_a, h_a, sizeof(cufftDoubleComplex)*NX*NY*NZ, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/* Create a 3D FFT plan. */
	if (cufftPlanMany(&plan, NRANK, n,
		NULL, 1, NX*NY*NZ, // *inembed, istride, idist
		NULL, 1, NX*NY*NZ, // *onembed, ostride, odist
		CUFFT_Z2Z, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		goto Error;
	}

	/* Use the CUFFT plan to transform the signal in place. */
	if (cufftExecZ2Z(plan, d_a, data, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
		goto Error;
	}



	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(result, data, sizeof(cufftDoubleComplex)*(NX*NY*NZ)*BATCH, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	//print result
	printf("CUFFT\n");
	count= 0;
	while(count<NX*NY*NZ){
		printf("%f + i %f\n",result[count].x, result[count].y);
		++count;}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaGetLastError();

Error:
	cufftDestroy(plan);
	cudaFree(data);
	cudaFree(d_a);

	return cudaStatus;
}








int main(int argc, char **argv){
	cufftDoubleComplex *data = new cufftDoubleComplex[NX*NY*NZ];
	cufftDoubleComplex *result;
	double  *cufft_output = new double[2*NX*NY*NZ];//fftw_input is real
  int count;

	result = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(NX*NY*NZ)*BATCH);
	data = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(NX*NY*NZ)*BATCH);

	//create small data cube inside larger data cube
	count = 0;
	for(int i=0;i<K;i++){
		for(int j=0;j<K;j++){
			for(int k=0;k<K;k++){
				data[NX*NY*i + NX*j + k ].x=i+j+k+0.3; //arbitrary value
				data[NX*NY*i + NX*j + k].y=0;
			}}}

			//pass same input to gpu. 'data' from host side.

			// Running cuFFT
			cudaError_t cudaStatus = PerfCuFFT(argc, argv, data, result);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "CuFFT failed!");
				return 1;
			}


			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				return 1;
			}


	delete [] data;
	delete [] result;
	return 0;
}
