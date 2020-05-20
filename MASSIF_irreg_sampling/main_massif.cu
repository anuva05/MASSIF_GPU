#include <iostream>
#include <string.h>
#include <math.h>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include "inputs.h"
#include "callbacks_octree.cu"
#include "helperfunctions.h"
#include "fftwfunctions.h"
#include "octree_table_host.cu"
/*******************************

 Main function

*********************************/





int main(int argc, char **argv){

  //Host variables

	cufftDoubleComplex *data;
	cufftDoubleComplex *small_cube;
	cufftDoubleComplex *result1;
	cufftDoubleComplex* unsampled_result;
  double *fftw_input = new double[2*NX*NY*NZ];
  double *fftw_output = new double[2*NX*NY*NZ];
	double  *cufft_output = new double[2*NX*NY*NZ];
  int count;
  int correct;
  fftw_plan plan3d[2];
  fftw_plan plan3dinv[2];

  //Device variables
  cufftDoubleComplex *d_result;
  cufftDoubleComplex *d_a;
  int final_samples;
	int blocks;
	int *octreeTable;
	int *d_octreeTable_temp;
	int XB, YB, ZB;
	int *ds_rates;
	void *px;
	XB = NX/OCTREE_FINEST;
	YB = NY/OCTREE_FINEST;
	ZB = NZ/OCTREE_FINEST;
	blocks = XB*YB*ZB;

  ds_rates= (int*)malloc(sizeof(int)*blocks);
	octreeTable = (int*)malloc(sizeof(int)*blocks*5);

  final_samples = octree_table_construct(ds_rates, octreeTable);

/*
  // trying to allocate struct

	 /// little experiment with the struct
	 CudaInput c(5);
	 // create class storage on device and copy top level class
	 CudaInput *d_c;

	 cudaMalloc((void **)&d_c, sizeof(CudaInput));
	 cudaMemcpy(d_c, &c, sizeof(CudaInput), cudaMemcpyHostToDevice);
	 // make an allocated region on device for use by pointer in class
	 int *temp_octree;
	 double *temp_result;


	 cudaMalloc((void **)&temp_octree, sizeof(int)*10);
	 cudaMemcpy(temp_octree, c.octree, sizeof(int)*10, cudaMemcpyHostToDevice);

	 cudaMalloc((void **)&temp_result, sizeof(double)*5);
	 cudaMemcpy(temp_result, c.result, sizeof(double)*5, cudaMemcpyHostToDevice);
	 // copy pointer to allocated device storage to device class
	 cudaMemcpy(&(d_c->octree), &temp_octree, sizeof(int *), cudaMemcpyHostToDevice);
	 cudaMemcpy(&(d_c->result), &temp_result, sizeof(double *), cudaMemcpyHostToDevice);



*/





	//allocating device side arrays
	cudaMalloc((void**)&d_a, sizeof(cufftDoubleComplex)*K*K*K);
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;
	}


	//Store the output samples in this array
	cudaMalloc((void**)&d_result, sizeof(cufftDoubleComplex)*final_samples);
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return 0;
	}



 	//allocating host side arrays. temporary use of uniform DS rate
	result1 = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(final_samples));
  unsampled_result=(cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*NX*NY*((NZ-K)/DS));
	data = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(NX*NY*NZ));
  small_cube = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(K*K*K));


  // Choosing CUDA device with newer architect
  //int dev = findCudaDevice(argc, (const char **)argv);





cout<<"creating data"<<endl;
//create small data cube inside larger data cube
count = 0;
for(int i=0;i<K;i++){
	for(int j=0;j<K;j++){
	  for(int k=0;k<K;k++){
	    data[NX*NY*i + NX*j + k ].x= i*j*k+0.3 ;//arbitrary value
	    data[NX*NY*i + NX*j + k].y=0;

	    small_cube[K*K*i + K*j + k].x = i*j*k + 0.3; //same value as data
	    small_cube[K*K*i + K*j + k].y=0;

	  }}}




  // Running cuFFT

  cout << "Run cufft" <<endl;
	auto start = high_resolution_clock::now();
  cudaError_t cudaStatus = minibatch_CuFFT(argc, argv, small_cube, result1, d_a, d_result, unsampled_result, &final_samples);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "CuFFT failed!");
    return 1;
  }
	auto stop = high_resolution_clock::now();
	auto durationCUDA = duration_cast<microseconds>(stop - start);

/*
  cout<< "copy result into double array"<< endl;
  //put result in cufft_output
  count = 0;
  for(int i=0;i<NZ;i++){
  for(int j=0;j<NY;j++){
  for(int k=0;k<NX;k++){
  cufft_output[count]= result[NX*NY*i + NX*j + k ].x;
  cufft_output[count+1] = result[NX*NY*i + NX*j + k].y;
  count = count + 2;
}}}

*/




cudaStatus = cudaDeviceReset();
if (cudaStatus != cudaSuccess) {
  fprintf(stderr, "cudaDeviceReset failed!");
  return 1;
}
/******************************
*
*
*  Create FFTW plan on CPU
*  and compute it for comparison
*******************************/

/*

printf("creating fftw plan\n");


start = high_resolution_clock::now();
create_3Dplan_forward(plan3d, fftw_input, fftw_output, NX, NY, NZ);
create_3Dplan_inverse(plan3dinv, fftw_input, fftw_output, NX, NY, NZ);


//input for fftw on cpu..convert data from double to fftw_complex format
cout<<"fftw input"<<endl;
count = 0;
for(int i=0;i<NZ;i++){
	for(int j=0;j<NY;j++){
		for(int k=0;k<NX;k++){
			fftw_input[count]= data[NX*NY*i + NX*j + k].x;
			fftw_input[count+1] = data[NX*NY*i + NX*j + k].y;
			count=count+2;
		}}}
cout<<"end of input, count="<< count << endl;

//execute fftw
cout<<"executing FFTW forward plan"<<endl;

execute_fftw_3d_plan_forward(plan3d, fftw_input, fftw_output);

//multiply by green's
count = 0;
for(int i=0;i<NZ;i++){
	for(int j=0;j<NY;j++){
		for(int k=0;k<NX;k++){
			fftw_input[count]= fftw_output[count]*2.0;
			fftw_input[count+1] = 0.0;
			count=count+2;
		}}}

cout<<"executing FFTW plan and printing output"<<endl;
execute_fftw_3d_plan_inverse(plan3dinv, fftw_input, fftw_output);


stop = high_resolution_clock::now();
auto durationFFTW = duration_cast<microseconds>(stop - start);



 if(TO_PRINT==1){


		 printResult(result, final_samples);

		 cout<< "CUFFT unsampled first plane"<<endl;
		 count = 0;
		 while(count<NX*NY){
			 cout<< count << ": CUFFT :" << unsampled_result[count].x <<"," << unsampled_result[count].y << endl;
			 count = count + 1;
		 }
		 cout<< "FFTW first plane"<<endl;
		 count = 0;
		 while(count<NX*NY){
			 cout<< count << ": FFTW:" << fftw_output[2*count] <<"," << fftw_output[2*count+1] << endl;
			 count = count + 1;
		 }

}
else{
 //output is too large, only print few values

		 cout<< "First few values of CUFFT output"<<endl;
		 count = 0;
		 while(count<20){
			 cout<< count << ": CUFFT:" << unsampled_result[count].x <<"," << unsampled_result[count].y << endl;
			 count = count + 1;
		 }
		 cout<< "First few values of FFTW output"<<endl;
		 count = 0;
		 while(count<20){
			 cout<< count << ": FFTW:" << fftw_output[2*count] <<"," << fftw_output[2*count+1] << endl;
			 count = count + 1;
		 }


}


//Print timing info
cout << "CUDA time duration (plan create + execute):" << double(durationCUDA.count())/1000000 << endl ;
cout << "FFTW time duration (plan create + execute):" << double(durationFFTW.count())/1000000 << endl ;

fftw_destroy_plan(*plan3d);
fftw_destroy_plan(*plan3dinv);


delete [] data;
delete [] result;
delete [] fftw_input;
delete [] fftw_output;
delete [] small_cube;
*/
return 0;

}
