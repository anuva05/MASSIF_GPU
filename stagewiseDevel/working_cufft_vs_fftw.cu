//Create data on cpu. Big cube with small cube inside
//Computes cufft and fftw and compares the answers
//whole N x N x N cube is materialized.


// includes, system

#include <iostream>
#include <string.h>
#include <math.h>
using namespace std;
// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <fftw3.h>
#include <float.h>
#define NX 4
#define NY 4
#define NZ 4
#define K 2  //dimension of small cube
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

void CreateData(cufftDoubleComplex *array){

	int count = 0;
	printf("data\n");	
	while (count< NX*NY*NZ){
		
		array[count].x = count; //real part
		array[count].y = 0;       //imag part
                printf("%f\n",array[count].x);
		++count;
	}
}

//some C functions to check the answer against fftw


extern "C" void create_3Dplan(fftw_plan *plan3d, double *temp, double *tempio, int m, int n, int k) {
  // full 3D plan
  fftw_iodim s0, s1[2], d0, d1;

  s0.n = k;
  s0.is = m * n;
  s0.os = m * n;

  d0.n = m * n;
  d0.is = 1;
  d0.os = 1;

  s1[0].n = m;
  s1[0].is = 1;
  s1[0].os = 1;

  s1[1].n = n;
  s1[1].is = m;
  s1[1].os = m;

  //d1.n = (k/2 + 1); //FOR R2C
  d1.n = k;
  d1.is = m * n;
  d1.os = m * n;

  *(plan3d + 0) = fftw_plan_guru_dft(1, &s0, 1, &d0, (fftw_complex *) tempio, (fftw_complex *) temp, FFTW_FORWARD, FFTW_MEASURE); //1 D fft
  *(plan3d + 1) = fftw_plan_guru_dft(2, s1, 1, &d1, (fftw_complex*) temp, (fftw_complex*) temp, FFTW_FORWARD, FFTW_MEASURE); // then 2d FFT
   if(*(plan3d+0)==NULL || *(plan3d + 1) ==NULL)
        printf("NULL PLAN");

}


extern "C" void execute_fftw_3d_plan(fftw_plan *plan3D, double *tempio0, double *temp) {

  int i;
  fftw_execute_dft(*(plan3D + 0), (fftw_complex*)tempio0, (fftw_complex*) temp);
  printf("temp output %lf \n", temp[1]);
  fftw_execute_dft(*(plan3D + 1), (fftw_complex*) temp, (fftw_complex*) temp);
  printf("FFTW output\n");
  for( i = 0; i != 2*(NX * NY * NZ); ++i) {
    printf("%lf\n", *(tempio0+ i)); 
	 }
}

extern "C" int verify_with_fftw(double *fftw_output, double *cufft_output){

  int correct = 1;
  int i ;
  for( i = 0; i != 2*(NX * NY * NZ); ++i) {
    printf("%lf %lf\n", *(fftw_output + i), *(cufft_output + i));
        if((fabs(*(fftw_output + i) - *(cufft_output + i)) > 1e-3) || (isnan(*(fftw_output + i))) || (isnan(*(cufft_output + i)))) {
              correct = 0;
                  }
                    }

        printf("Correctness: %d\n", correct);
        return correct;
}






int main(int argc, char **argv){
	cufftDoubleComplex *data = new cufftDoubleComplex[NX*NY*NZ];
	cufftDoubleComplex *result;
	double *fftw_input = new double[2*NX*NY*NZ];
	double *fftw_output = new double[2*NX*NY*NZ];
	double  *cufft_output = new double[2*NX*NY*NZ];//fftw_input is real	
        int count;
	int correct;
	fftw_plan plan3d[2];
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



	//Creat FFTW plan on CPU
	printf("creating fftw plan\n");
        create_3Dplan(plan3d, fftw_input, fftw_output, NX, NY, NZ);
	
	//input for fftw on cpu..convert data from double to fftw_complex format
	printf("fftw input");
	count = 0;
	for(int i=0;i<NZ;i++){
	 for(int j=0;j<NY;j++){
	  for(int k=0;k<NX;k++){
	    fftw_input[count]= data[NX*NY*i + NX*j + k].x;
	    fftw_input[count+1] = data[NX*NY*i + NX*j + k].y;
	    printf("%lf + i %lf\n", fftw_input[count],fftw_input[count+1]);   
	    count=count+2;
	  }}}
	printf("end of input, count = %d", count);

	 //pass same input to gpu. 'data' from host side.

	// Running cuFFT
	cudaError_t cudaStatus = PerfCuFFT(argc, argv, data, result);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CuFFT failed!");
		return 1;
	}

	//put result in cufft_output
        count = 0;
        for(int i=0;i<NZ;i++){
         for(int j=0;j<NY;j++){
          for(int k=0;k<NX;k++){
            cufft_output[count]= result[NX*NY*i + NX*j + k ].x;
	    cufft_output[count+1] = result[NX*NY*i + NX*j + k].y;
	    count = count + 2;
          }}}


	//copy  to cpu


	//execute fftw 
	cout<<"executing plan"<<endl;
        execute_fftw_3d_plan(plan3d, fftw_input, fftw_output);
        printf("checking correctness\n");
	correct = verify_with_fftw(fftw_output, cufft_output);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}


	delete [] data;
	delete [] result;
	return 0;
}
