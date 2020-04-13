#include <iostream>
#include <string.h>
#include <math.h>
using namespace std;
#include "inputs.h"
#include "callbacks.cu"
#include "helperfunctions.h"
#include "fftwfunctions.h"
/*******************************

 Main function

*********************************/





int main(int argc, char **argv){

  //Host variables
	cufftDoubleComplex *data;
	cufftDoubleComplex *small_cube;
	cufftDoubleComplex *result;
  double *fftw_input = new double[2*NX*NY*NZ];
  double *fftw_output = new double[2*NX*NY*NZ];
	double  *cufft_output = new double[2*NX*NY*NZ];
  int count,k,count1,count2;
  int idx, orig_idx;
  int correct;
  fftw_plan plan3d[2];
  fftw_plan plan3dinv[2];
  int x, y, z;
  //Device variables
  cufftDoubleComplex *d_result;//FULL N^3
  cufftDoubleComplex *d_a; //small cube K x K x K (technically real values)
  int final_samples;
  final_samples =  (K*K + (NX*NY - K*K)/(DS*DS))*K + (NX*NY/(DS*DS))*(NZ-K)/DS;

  cufftDoubleComplex* unsampled_result;

 	//allocating host side arrays
	result = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(final_samples));
  unsampled_result=(cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*NX*NY*((NZ-K)/DS));
	data = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(NX*NY*NZ));
  small_cube = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(K*K*K));


  // Choosing CUDA device with newer architect
  //int dev = findCudaDevice(argc, (const char **)argv);

  //allocating device side arrays
  cudaMalloc((void**)&d_a, sizeof(cufftDoubleComplex)*K*K*K);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
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
        data[NX*NY*i + NX*j + k ].x= i*j*k+0.3 ;//arbitrary value
        data[NX*NY*i + NX*j + k].y=0;

        small_cube[K*K*i + K*j + k].x = i*j*k + 0.3; //same value as data
        small_cube[K*K*i + K*j + k].y=0;

      }}}


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



  // Running cuFFT

  cout << "Run cufft" <<endl;
  cudaError_t cudaStatus = minibatch_CuFFT(argc, argv, small_cube, result, d_a, d_result, unsampled_result);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "CuFFT failed!");
    return 1;
  }


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
*
*******************************/

printf("creating fftw plan\n");
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

/*
   //Printing some values
   cout<<"Domain values - first 4 planes "<<endl;
   count = 0;
   while(count<NX*NY*4){

      x= count % NX;
      y= (count/NX)%NY;
      z =(count/(NX*NY));
      cout<< count << ":" << "indices:" << x <<","<<y<<"," <<z <<":  " << result[count].x << ", " <<result[count].y << "\t FFTW:" << fftw_output[2*count] <<"," << fftw_output[2*count+1] << endl;
      count = count + 1;
    }

    //cout<<"CUfft Samples in first plane"<<endl;
    count1 =0;
    count2 = 0;
    // while(count1< (K*K + (NX*NY-K*K)/(DS*DS) ) ){
    //    cout<< count1 << ":" << result[count1 ].x << ", " <<result[count1].y  << endl;
    //    count1 = count1 + 1;
    //  }
    //
*/

  printResult(result, final_samples);
/*
       for(int j=0;j<NY;j++){
         for(int k=0;k<NX;k++){

           if((k>=0 && k <K)&&(j>=0 && j <K)){

             orig_idx = NX*j + k;
              printf("idx: %d, val = %lf, %lf\n", orig_idx, result[ K*j + k].x,result[ K*j + k].y);
           }

           if((k%DS == 0 && j%DS == 0)&&(k>=K || j >=K)) {

             idx = (NX/DS)*(j/DS) + k/DS - (NX*NY-K*K)/(DS*DS);
              orig_idx = NX*j + k;
              printf("idx: %d, val = %lf, %lf\n", orig_idx, result[K*K + idx].x,result[K*K + idx ].y);
           }

         }}

*/

 if(TO_PRINT==1){

cout<< "cufft unsampled first plane"<<endl;
count2 = 0;
 while(count2<NX*NY){
   cout<< count2 << ": cufft:" << unsampled_result[count2].x <<"," << unsampled_result[count2].y << endl;
   count2 = count2 + 1;
 }
     cout<< "FFTW first plane"<<endl;
     count2 = 0;
      while(count2<NX*NY){
        cout<< count2 << ": FFTW:" << fftw_output[2*count2] <<"," << fftw_output[2*count2+1] << endl;
        count2 = count2 + 1;
      }
    // count = 0;
    // while(count<final_samples){
    //    cout<< count << ":" << result[count].x << ", " <<result[count].y  << endl;
    //    count = count + 1;
    //  }
}


        fftw_destroy_plan(*plan3d);
        fftw_destroy_plan(*plan3dinv);


        delete [] data;
        delete [] result;
        delete [] fftw_input;
        delete [] fftw_output;
        delete [] small_cube;
        return 0;
}
