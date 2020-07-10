#include <iostream>
#include <string.h>
#include <math.h>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include "inputs.h"
#include "callbacks.cu"
#include "helperfunctions.h"
/*******************************

 Main function

*********************************/





int main(int argc, char **argv){

  //Host variables
  cufftDoubleComplex *small_cube;
  cufftDoubleComplex *result;
  int count;
  int correct;

  //Device variables
  cufftDoubleComplex *d_result;
  cufftDoubleComplex *d_a;
  int final_samples;
  

  final_samples =  (K*K + (NX*NY - K*K)/(DS*DS))*K + (NX*NY/(DS*DS))*(NZ-K)/DS;

  cufftDoubleComplex* unsampled_result;

 	//allocating host side arrays
  result = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(final_samples));
  unsampled_result=(cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*NX*NY*((NZ-K)/DS));
  small_cube = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(K*K*K));

  printf("here\n");
  // Choosing CUDA device with newer architect
  //int dev = findCudaDevice(argc, (const char **)argv);

  //allocating device side arrays
  cudaMalloc((void**)&d_a, sizeof(cufftDoubleComplex)*K*K*K);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate small cube\n");
    return 0;
  }


  //**TEMPORARY** the output is going to materialize the full cube for simplicity
  cudaMalloc((void**)&d_result, sizeof(cufftDoubleComplex)*final_samples);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return 0;
  }




cout<<"creating data"<<endl;
//create small data cube inside larger data cube
count = 0;
for(int i=0;i<K;i++){
	for(int j=0;j<K;j++){
	  for(int k=0;k<K;k++){

	    small_cube[K*K*i + K*j + k].x = i*j*k + 0.3; //same value as data
	    small_cube[K*K*i + K*j + k].y=0;

	  }}}

/*

for(int i=0;i<K;i++){
  for(int j=0;j<K;j++){
    for(int k=0;k<K;k++){
      cout<< "data " << data[NX*NY*i + NX*j + k ].x << endl;

    }}}

    for(int i=0;i<K;i++){
      for(int j=0;j<K;j++){
        for(int k=0;k<K;k++){

          cout<< "small cube " <<small_cube[K*K*i + K*j + k].x << endl;


        }}}
*/


  // Running cuFFT

  cout << "Run cufft" <<endl;
  
  auto start = high_resolution_clock::now();
  cudaError_t cudaStatus = minibatch_CuFFT(argc, argv, small_cube, result, d_a, d_result, unsampled_result);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "CuFFT failed!");
    return 1;
  }
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);


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





 if(TO_PRINT==1){


		 printResult(result, final_samples);

		 cout<< "CUFFT unsampled first plane"<<endl;
		 count = 0;
		 while(count<NX*NY){
			 cout<< count << ": CUFFT :" << unsampled_result[count].x <<"," << unsampled_result[count].y << endl;
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


}


cout << double(duration.count())/1000000 << "Seconds"  << endl;



delete [] result;
delete [] small_cube;
return 0;
}
