//single precision
// for the first time, integrating the whole pipeline of forward fft, conv and inverse fft
//not implementing sampling
#include <iostream>
#include <string.h>
#include <math.h>
using namespace std;

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <fftw3.h>
#include <float.h>
//answer with fftw matches for 1D and 2D (NX=8, NY=NZ=1 or NX=8, NY=8, NZ=1)
#define NX 4
#define NY 4
#define NZ 4
#define K 2  //dimension of small cube
#define B 8 // Number of pencils in one batch
#define NRANK 3
# define DS 1 //downsample rate
//need to define GPU constants for s0, c0

/* Callback functions for padding. Callbacks are element-wise */


__device__ cufftComplex pad_stage0(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {

   // K x K x K is padded to NX x NY x K
    cufftComplex *input = (cufftComplex*) dataIn;
    int dim0 = ((offset) % NX) - 0;
    int dim1 = ((offset / NX) % NY) - 0;
    int dim2 = ((offset / (NX*NY)) ) - 0;

    //printf("padstage0 dims= %d,%d,%d\n", dim0, dim1, dim2);
     cufftComplex r;

   if (((0 <= dim0) && (dim0 < K)) && ((0 <= dim1) && (dim1 < K)) && ((0 <= dim2) && (dim2 < K))) {
     //printf("offset:%d, val:%lf\n", offset, (*(input + ((1 * dim0) + (K * dim1) + (K*K * dim2)))).x);
      return *(input + ((1 * dim0) + (K* dim1) + (K*K * dim2)));
   }
  else {
     //printf("offset:%d, val: 0 \n", offset);
       r.x = 0.0;
       r.y = 0.0;
       return r;
   }
}
__device__ cufftCallbackLoadC d_pad_stage0 = pad_stage0;

//load callback for next stage
__device__ cufftComplex pad_stage1(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {

   // B pencils of length K are padded to B pencils of length NZ.
   //Input layout has x has fastest varying dimension, followed by y, then z.
   //idist = 1, istride = B
   //hence, if offset < B*K, then return value. Else return 0
    cufftComplex *input = (cufftComplex*) dataIn;
    cufftComplex r;

   if (offset<B*K) {
      r =*(input + offset);
      printf("offset:%d, val: %lf \n", offset, r.x);
      return r;
   }
  else {

       r.x = 0.0;
       r.y = 0.0;
       printf("offset:%d, val: 0 \n", offset);
       return r;
   }
}
__device__ cufftCallbackLoadC d_pad_stage1 = pad_stage1;

//store callback
__device__ void greens_pointwise (void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr) {

    cufftComplex r;
    r.x= 2.0*element.x;
    r.y= 0.0;

	  ((cufftComplex*)dataOut)[offset] = r;
}

//to load the callback function onto the CPU, since it is a device function and resides only on GPU
__device__ cufftCallbackStoreC d_greens_pointwise = greens_pointwise;




/* Helper function performing Cufft */

cudaError_t minibatch_CuFFT(int argc, char **argv, cufftComplex* h_a, cufftComplex* result, cufftComplex* d_a, cufftComplex* d_result){
	cudaError_t cudaStatus;
  cufftResult cufftStatus;
	cufftHandle *plans;
	int count, offset;
	int b;
//	long *input_address, *output_address;
	 cufftComplex* d_fw_stage0;
   cufftComplex* d_temp1;
   cufftComplex* d_temp2; //to hold temporary group of B pencils of length K from the slab
   cufftComplex* d_temp3;
   cufftComplex* d_fw_stage1;
   cufftComplex* d_inv_stage1;
   int i,strideIdx;
	 cufftComplex* t;


	plans = (cufftHandle*) malloc(4 * sizeof(cufftHandle));


  // for the first plan
	 int rank = 2;
   int batch=K;
   int t_size[] = {NX,NY};
   int inembed0[] = {NX,NY};
   int istride = 1;
   int idist= NX*NY;
   int odist = NX*NY;//how far next output signal is from current
   int onembed0[] ={NX, NY};
   int ostride = 1;
   int *d_offset;
 //for second plan
   int inembed1 =  B*NZ;// after padding
   int onembed1 =  B*NZ;// after convolution of all pencils
   int n_1 =NZ;
   //for third plan ifft group of B pencils
   int inembed2 =  B*NZ;///??
   int onembed2 =  B*NZ; //
   int n_2 =NZ;


    //----for fourth plan----//
   int inembed3[] =   {NX,NY};
   int onembed3[] = {NX,NY}; // {K*K + (NX*NY-K*K)/(DS*DS)}; //??
   int n_3[] = {NX,NY};
   int num_samples;


	cudaMalloc((void**)&d_offset, sizeof(int));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
  }

   cudaMalloc((void**)&d_fw_stage0, sizeof(cufftComplex)*(NX*NY*K));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    exit(-1);
  }

  cudaMalloc((void**)&d_temp1, sizeof(cufftComplex)*B*K);
if (cudaGetLastError() != cudaSuccess){
  fprintf(stderr, "Cuda error: Failed to allocate\n");
  exit(-1);
}
cudaMalloc((void**)&d_temp2, sizeof(cufftComplex)*B*NZ);
if (cudaGetLastError() != cudaSuccess){
fprintf(stderr, "Cuda error: Failed to allocate\n");
exit(-1);
}
cudaMalloc((void**)&d_temp3, sizeof(cufftComplex)*B*NZ);
if (cudaGetLastError() != cudaSuccess){
fprintf(stderr, "Cuda error: Failed to allocate\n");
exit(-1);
}



    cudaMalloc((void**)&d_fw_stage1, sizeof(cufftComplex)*B*NZ);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    exit(-1);
  }

    cudaMalloc((void**)&d_inv_stage1, sizeof(cufftComplex)*NX*NY*NZ);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    exit(-1);
  }



  //copy data
	 cudaStatus = cudaMemcpy(d_a, h_a, sizeof(cufftComplex)*K*K*K, cudaMemcpyHostToDevice);
          if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "dev_in cudaMalloc failed!");
                 exit(-1);
        }



	cout<<"creating first stage fft plan"<<endl;
	if (cufftPlanMany((plans+0), rank, t_size, inembed0, istride, idist, onembed0, ostride, odist, CUFFT_C2C, batch)!=CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		goto Error;
	};

	//Set callback function to do zero padding. Load callback function and attach it to plan_0
	cufftCallbackLoadC h_pad_stage0;
  cudaStatus = cudaMemcpyFromSymbol(&h_pad_stage0, d_pad_stage0, sizeof(h_pad_stage0));
  if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "cudamemcpyfromsymbol failed!");
                 exit(-1);
        }


	cufftStatus = cufftXtSetCallback(*(plans+0), (void**)&h_pad_stage0, CUFFT_CB_LD_COMPLEX, NULL);
  cout<< "cufftstatus:" << cufftStatus <<endl;

	cout<<"creating second stage plan"<<endl;

	//Create second plan that computes a batch of B pencils in each execution
  if (cufftPlanMany((plans+1), 1, &n_1,
                        &inembed1, B, 1, // *inembed, istride, idist
                        &onembed1, B, 1, // *onembed, ostride, odist
                        CUFFT_C2C, B) != CUFFT_SUCCESS){
                     fprintf(stderr, "CUFFT error: Plan creation failed");
                     goto Error;
                }

	//set callback functions for plan_1.
  //Load callback: Padding in Z dimension
  //Store callback: performing pointwise multiplication

cufftCallbackLoadC h_pad_stage1;
cudaMemcpyFromSymbol(&h_pad_stage1, d_pad_stage1, sizeof(h_pad_stage1));
cufftXtSetCallback(*(plans+1),(void **)&h_pad_stage1,CUFFT_CB_LD_COMPLEX,(void**)&d_offset);

cufftCallbackStoreC h_greens_pointwise;
cudaMemcpyFromSymbol(&h_greens_pointwise, d_greens_pointwise, sizeof(h_greens_pointwise));
cufftXtSetCallback(*(plans+1),(void **)&h_greens_pointwise,CUFFT_CB_ST_COMPLEX,(void**)&d_offset);
cudaDeviceSynchronize();

//invert the batch of B pencils.

if (cufftPlanMany((plans+2), 1, &n_2,
&inembed2, B, 1, // *inembed, istride, idist
&onembed2, B, 1, // *onembed, ostride, odist
CUFFT_C2C, B) != CUFFT_SUCCESS){
  fprintf(stderr, "CUFFT error: Plan creation failed");
  goto Error;
}
// perform inverse transform in X and Y and sample

if (cufftPlanMany((plans+3), 2, n_3,
inembed3, 1, NX*NY, // *inembed, istride, idist
onembed3, 1,NX*NY,  // *onembed, ostride, odist|Previously ostride = 1, odist =NX*NY; We store the output as B pencils of size N
CUFFT_C2C, NZ) != CUFFT_SUCCESS){
  fprintf(stderr, "CUFFT error: Plan creation failed");
  goto Error;
}

//-------------------------  execute plans ---------------------------//
cout<<"executing first stage fft"<<endl;


	 if (cufftExecC2C(*(plans+0), d_a, d_fw_stage0, CUFFT_FORWARD) != CUFFT_SUCCESS){
                        fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
                        goto Error;
                }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    goto Error;
  }



	//next step - transform in Z using tiling and minibatches
	for(b=0;b<(NX*NY)/B ;b++){

    		cout<<"Executing batch number"<< b << endl;
    		offset = (b*B); //each batch processes B pencils
        cout<< "Offset:" << offset << endl;

        //the device variable being copied to should be a pointer..and pas address of host side int
    		cudaStatus = cudaMemcpy(d_offset, &offset, sizeof(int), cudaMemcpyHostToDevice);
              if (cudaStatus != cudaSuccess) {
                     fprintf(stderr, "d_offset cudaMalloc failed!");
                     exit(-1);
            }

       cudaDeviceSynchronize();

      //copy B x K group of pencils into d_temp
      //ctr =0
      for(strideIdx =0; strideIdx< K; strideIdx++){
        cudaStatus = cudaMemcpy(d_temp1+strideIdx*B, d_fw_stage0+offset+strideIdx*NX*NY, sizeof(cufftComplex)*B, cudaMemcpyDeviceToDevice);
        if (cudaStatus != cudaSuccess) {
               fprintf(stderr, "Pencils cudaMalloc failed!");
               exit(-1);
             }
      }




    	if (cufftExecC2C(*(plans+1), d_temp1, d_temp2, CUFFT_FORWARD) != CUFFT_SUCCESS){
    			fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
    			goto Error;
    		}


    	cudaStatus = cudaDeviceSynchronize();
      if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                    goto Error;
            }


       //Perform ifft
       if (cufftExecC2C(*(plans+2), d_temp2, d_temp3, CUFFT_INVERSE) != CUFFT_SUCCESS){
         fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
         goto Error;
       }

       //put the result in the appropriate place
       for(strideIdx =0; strideIdx< NZ; strideIdx++){
         cudaStatus = cudaMemcpy(d_inv_stage1+offset+strideIdx*NX*NY, d_temp3+strideIdx*B, sizeof(cufftComplex)*B, cudaMemcpyDeviceToDevice);
         if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Post FFT cudaMalloc failed!");
                exit(-1);
              }
       }


      cudaStatus = cudaDeviceSynchronize();
      if (cudaStatus != cudaSuccess) {
                   fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                   goto Error;
           }


	}//all batches processed

// //  Last stage, 2D inverse transform with sampling
   if (cufftExecC2C(*(plans+3), d_inv_stage1, d_result, CUFFT_INVERSE) != CUFFT_SUCCESS){
     fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
     goto Error;
   }
 cudaStatus = cudaDeviceSynchronize();
       if (cudaStatus != cudaSuccess) {
               fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
               goto Error;
       }
//
//  //Copy out output
//
  num_samples =  NX*NY*NZ; //(K + (NX-K)/DS)*(K + (NY-K)/DS)*(K + (NZ-K)/DS);
 	cudaStatus = cudaMemcpy(result, d_result, sizeof(cufftComplex)*(num_samples), cudaMemcpyDeviceToHost);
 	if (cudaStatus != cudaSuccess){
 		fprintf(stderr, "cudaMemcpy failed!");
 		goto Error;
 	cout<<"CUFFT output (First few values)"<<endl;
  }

  count= 0;
 	while(count< num_samples){
 		cout<<count << " " << result[count].x << "," << result[count].y<<endl;
 		++count;}

 	cudaStatus = cudaDeviceSynchronize();
 	if (cudaStatus != cudaSuccess) {
 		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
 		goto Error;
 	}

 	cudaStatus = cudaGetLastError();

Error:
	//Delete the CUFFT Plan
	cufftDestroy(*(plans + 0));
  cufftDestroy(*(plans + 1));
  cufftDestroy(*(plans + 2));
  cufftDestroy(*(plans + 3));
	free(plans);
	cudaFree(d_a);
  cudaFree(d_temp1);
  cudaFree(d_temp2);
	cudaFree(d_fw_stage0);
	cudaFree(d_fw_stage1);
	cudaFree(d_inv_stage1);
	cudaFree(d_result);
	cudaFree(d_offset);
	return cudaStatus;

}

//fftw functions

extern "C" void create_3Dplan_forward(fftwf_plan *plan3d, float *temp, float *tempio, int m, int n, int k) {
  // full 3D plan
  fftwf_iodim s0, s1[2], d0, d1;

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

  *(plan3d + 0) = fftwf_plan_guru_dft(1, &s0, 1, &d0, (fftwf_complex *) tempio, (fftwf_complex *) temp, FFTW_FORWARD, FFTW_MEASURE); //1 D fft
  *(plan3d + 1) = fftwf_plan_guru_dft(2, s1, 1, &d1, (fftwf_complex*) temp, (fftwf_complex*) temp, FFTW_FORWARD, FFTW_MEASURE); // then 2d FFT
  if(*(plan3d + 0)==NULL || *(plan3d + 1)==NULL)
  printf("NULL PLAN");

}
extern "C" void create_3Dplan_inverse(fftwf_plan *plan3dinv, float *temp, float *tempio, int m, int n, int k) {
  // full 3D plan
  fftwf_iodim s0[2], s1, d0, d1;



  s0[0].n = m;
  s0[0].is = 1;
  s0[0].os = 1;

  s0[1].n = n;
  s0[1].is = m;
  s0[1].os = m;

  d0.n = k;
  d0.is = m * n;
  d0.os = m * n;

  s1.n = k;
  s1.is = m * n;
  s1.os = m * n;

  d1.n = m * n;
  d1.is = 1;
  d1.os = 1;

  *(plan3dinv + 0) = fftwf_plan_guru_dft(2, s0, 1, &d0, (fftwf_complex *) tempio, (fftwf_complex *) temp, FFTW_BACKWARD, FFTW_MEASURE); //1 D fft
  *(plan3dinv + 1) = fftwf_plan_guru_dft(1, &s1, 1, &d1, (fftwf_complex*) temp, (fftwf_complex*) temp, FFTW_BACKWARD, FFTW_MEASURE); // then 2d FFT
  if(*(plan3dinv+0)==NULL || *(plan3dinv + 1) ==NULL)
  printf("NULL PLAN");

}


extern "C" void execute_fftwf_3d_plan_forward(fftwf_plan *plan3D, float *tempio0, float *temp) {


  fftwf_execute_dft(*(plan3D + 0), (fftwf_complex*)tempio0, (fftwf_complex*) temp);
  fftwf_execute_dft(*(plan3D + 1), (fftwf_complex*) temp, (fftwf_complex*) temp);

}
extern "C" void execute_fftwf_3d_plan_inverse(fftwf_plan *plan3Dinv, float *tempio0, float *temp) {
  int i;
  fftwf_execute_dft(*(plan3Dinv + 0), (fftwf_complex*)tempio0, (fftwf_complex*) temp);
  fftwf_execute_dft(*(plan3Dinv + 1), (fftwf_complex*) temp, (fftwf_complex*) temp);
   i= 0;
  printf("FFTW output (first few values)\n");
  while(i<2*NX*NY*NZ){
     printf("%d:, %lf, %lf\n", i, *(temp+ i), *(temp+i+1));
     i= i + 2;
  }

}


extern "C" int verify_with_fftw(float *fftwf_output, float *cufft_output){

  int correct = 1;
  int i ;
  for( i = 0; i != 2*(NX * NY * NZ); ++i) {
    printf("Index: %d, FFTW: %lf, CUFFT: %lf\n",i, *(fftwf_output + i), *(cufft_output + i));
        if((fabs(*(fftwf_output + i) - *(cufft_output + i)) > 1e-3) || (isnan(*(fftwf_output + i))) || (isnan(*(cufft_output + i)))) {
              correct = 0;
                  }
                    }

        printf("Correctness: %d\n", correct);
        return correct;
}




/*******************************

 Main function

*********************************/





int main(int argc, char **argv){

  //Host variables
	cufftComplex *data;
	cufftComplex *small_cube;
	cufftComplex *result;
  float *fftwf_input = new float[2*NX*NY*NZ];
  float *fftwf_output = new float[2*NX*NY*NZ];
	float  *cufft_output = new float[2*NX*NY*NZ];
  int count;
  int correct;
  fftwf_plan plan3d[2];
  fftwf_plan plan3dinv[2];

  //Device variables
  cufftComplex *d_result;//FULL N^3
  cufftComplex *d_a; //small cube K x K x K (technically real values)
  int final_samples;
  final_samples =   NX*NY*NZ; // ( K + (NX-K)/DS )*(K + (NY-K)/DS)*(K + (NZ-K)/DS);;


 	//allocating host side arrays
	result = (cufftComplex*)malloc(sizeof(cufftComplex)*(final_samples));
	data = (cufftComplex*)malloc(sizeof(cufftComplex)*(NX*NY*NZ));
  small_cube = (cufftComplex*)malloc(sizeof(cufftComplex)*(K*K*K));


  // Choosing CUDA device with newer architect
  //int dev = findCudaDevice(argc, (const char **)argv);

  //allocating device side arrays
  cudaMalloc((void**)&d_a, sizeof(cufftComplex)*K*K*K);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return 0;
  }


  //**TEMPORARY** the output is going to materialize the full cube for simplicity
  cudaMalloc((void**)&d_result, sizeof(cufftComplex)*final_samples);
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
        data[NX*NY*i + NX*j + k ].x= i*j*k+0.1 ;//arbitrary value
        data[NX*NY*i + NX*j + k].y=0;

        small_cube[K*K*i + K*j + k].x = i*j*k + 0.1; //same value as data
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
  cudaError_t cudaStatus = minibatch_CuFFT(argc, argv, small_cube, result, d_a, d_result);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "CuFFT failed!");
    return 1;
  }


/*
  cout<< "copy result into float array"<< endl;
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
create_3Dplan_forward(plan3d, fftwf_input, fftwf_output, NX, NY, NZ);
create_3Dplan_inverse(plan3dinv, fftwf_input, fftwf_output, NX, NY, NZ);


//input for fftw on cpu..convert data from float to fftwf_complex format
cout<<"fftw input"<<endl;
count = 0;
for(int i=0;i<NZ;i++){
  for(int j=0;j<NY;j++){
    for(int k=0;k<NX;k++){
      fftwf_input[count]= data[NX*NY*i + NX*j + k].x;
      fftwf_input[count+1] = data[NX*NY*i + NX*j + k].y;
      count=count+2;
    }}}
    cout<<"end of input, count="<< count << endl;

    //execute fftw
    cout<<"executing FFTW forward plan"<<endl;
    execute_fftwf_3d_plan_forward(plan3d, fftwf_input, fftwf_output);


    //multiply by green's
    count = 0;
    for(int i=0;i<NZ;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NX;k++){
          fftwf_input[count]= fftwf_output[count]*2.0;
          fftwf_input[count+1] = 0.0;
          count=count+2;
        }}}

        cout<<"executing FFTW plan and printing output"<<endl;
        execute_fftwf_3d_plan_inverse(plan3dinv, fftwf_input, fftwf_output);



        fftwf_destroy_plan(*plan3d);
        fftwf_destroy_plan(*plan3dinv);


        delete [] data;
        delete [] result;
        delete [] fftwf_input;
        delete [] fftwf_output;
        delete [] small_cube;
        return 0;
}
