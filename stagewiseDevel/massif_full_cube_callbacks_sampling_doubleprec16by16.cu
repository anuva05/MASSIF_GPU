//single precision
// for the first time, integrating the whole pipeline of forward fft, conv and inverse fft
//Remove the samples that we want to leave out, using Callbacks
//This approach saves memory
//Single precision = cufft and fftw will differ slightly for small value

//Important addition: the last IFFT is in place. The attached store callback function
//takes in a pointer to d_result from the callerInfo argument. d_result is of a custom size
//that is determined by downsampling. we choose to keep some elements of the ifft output, and these
//go in d_result. The others are not stored.
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
#define NX 16
#define NY 16
#define NZ 16
#define K 4  //dimension of small cube
#define B 16 // Number of pencils in one batch
#define NRANK 3
# define DS 4//downsample rate
//need to define GPU constants for s0, c0

/* Callback functions for padding. Callbacks are element-wise */


__device__ cufftDoubleComplex pad_stage0(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {

   // K x K x K is padded to NX x NY x K
    cufftDoubleComplex *input = (cufftDoubleComplex*) dataIn;
    int dim0 = ((offset) % NX) - 0;
    int dim1 = ((offset / NX) % NY) - 0;
    int dim2 = ((offset / (NX*NY)) ) - 0;

    //printf("padstage0 dims= %d,%d,%d\n", dim0, dim1, dim2);
     cufftDoubleComplex r;

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
__device__ cufftCallbackLoadZ d_pad_stage0 = pad_stage0;

//load callback for next stage
__device__ cufftDoubleComplex pad_stage1(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {

   // B pencils of length K are padded to B pencils of length NZ.
   //Input layout has x has fastest varying dimension, followed by y, then z.
   //idist = 1, istride = B
   //hence, if offset < B*K, then return value. Else return 0
    cufftDoubleComplex *input = (cufftDoubleComplex*) dataIn;
    cufftDoubleComplex r;

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
__device__ cufftCallbackLoadZ d_pad_stage1 = pad_stage1;

//store callback
__device__ void greens_pointwise (void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo, void *sharedPtr) {

    cufftDoubleComplex r;
    r.x= 2.0*element.x;
    r.y= 0.0;

	  ((cufftDoubleComplex*)dataOut)[offset] = r;
}

//to load the callback function onto the CPU, since it is a device function and resides only on GPU
__device__ cufftCallbackStoreZ d_greens_pointwise = greens_pointwise;


/* --------- Callbacks that perform sampling -------------*/

__device__ void sample_stage0(void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo, void *sharedPtr) {

//if 'z' value is not in 0 to K or not one of the pre-specified planes, then set it to 0.
  
  int z;
  int el;
  z = offset/B;
  el = offset%B;
  if (z>=0 && z<K){
    ((cufftDoubleComplex*)dataOut)[offset] = element;
  }
  //If the point is outside the domain but divisible,
  //we store the domain contiguously and the blocks corresponding
  //to the z values to keep are placed next
  //Size of stored output = B*(K + (NZ-K)/DS)
  if ((z%DS== 0)&&(z>=K)){
    ((cufftDoubleComplex*)dataOut)[B*(K + (z/DS)-1) + el]= element;
  }
}
__device__ cufftCallbackStoreZ d_sample_stage0 = sample_stage0;



__device__ void sample_stage1(void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo, void *sharedPtr) {

  // X and Y values should be corresponding to the samples we want to keep
  int dim0, dim1, dim2;
  int off = static_cast<int>(offset);
  int numSamplesPerPlane, samplesPerXRow, numRepeatedSamples;
  int idx;//special index for samplem
  cufftDoubleComplex *d_result = (cufftDoubleComplex*)callerInfo;


  dim2 = ( off)/(NX*NY)  ;
  dim1 = (off - NX*NY*dim2)/NX;
  dim0 = (off - NX*NY*dim2 - NX*dim1);

  numSamplesPerPlane = K*K + (NX*NY - K*K)/(DS*DS);
  samplesPerXRow = NX/DS;
  numRepeatedSamples = (K*K)/(DS*DS);//in first K planes


  //Put the element in the same place for the cufft output array - no change here
  ((cufftDoubleComplex*)dataOut)[offset] = element;


  //create custom array to store samples
  //if point is in domain
  if(dim2<K){
    if((dim0>=0 && dim0 <K)&&(dim1>=0 && dim1 <K)){
      //store in appropriate location in the compressed output
      d_result[numSamplesPerPlane*dim2 + K*dim1 + dim0] =element;
    }

    if((dim0>=K || dim1 >=K)&&(dim0%DS == 0 && dim1%DS == 0) ){
      idx = samplesPerXRow*(dim1/DS) + dim0/DS - numRepeatedSamples;
      d_result[numSamplesPerPlane*dim2 + K*K + idx ] =element;
    }

  }

  if(dim2>=K){

    if(dim0%DS == 0 && dim1%DS == 0){
      idx = samplesPerXRow*(dim1/DS) + dim0/DS;
      d_result[numSamplesPerPlane*K + samplesPerXRow*samplesPerXRow*(K-dim2) + idx ] =element;

    }
  }
}



__device__ cufftCallbackStoreZ d_sample_stage1 = sample_stage1;


/* --------- Callbacks that perform sampling End -------------*/

void print_XYplanes( cufftDoubleComplex* arr,int X, int Y, int Z){

  int i,j,k;
  for(k=0; k<Z; k++){
    printf("Plane %d:\n", k );
    for(j=0;j<Y; j++){
      for(i=0;i<X;i++){
        printf("Idx %d: val= %lf, %lf\n", X*Y*k + X*j + i, arr[X*Y*k + X*j + i ].x ,arr[X*Y*k + X*j + i ].y   );

      }
    }
  }

}//PRINT PLANES FUNCTION

//function to print
void printResult(cufftDoubleComplex *result, int samples){
int dim0, dim1, dim2;
int numSamplesPerPlane, samplesPerXRow, numRepeatedSamples;
int idx;//special index for samplem
int offset;

offset= 0;

while(offset< NX*NY*(K + (NZ-K)/DS) ){

  dim2 = ( offset) / (NX*NY)  ;

  dim1 = (offset - NX*NY*dim2)/NX;
  dim0 = (offset - NX*NY*dim2 - NX*dim1);
  numSamplesPerPlane = K*K + (NX*NY - K*K)/(DS*DS);
  samplesPerXRow = NX/DS;
  numRepeatedSamples = (K*K)/(DS*DS);

  if(dim2<K){
    if((dim0>=0 && dim0 <K)&&(dim1>=0 && dim1 <K)){
      //store in appropriate location in the compressed output

      printf("Offset: %d, val: %lf, %lf \n", offset, result[numSamplesPerPlane*dim2 + K*dim1 + dim0].x, result[numSamplesPerPlane*dim2 + K*dim1 + dim0].y);

    }

    else{
      if((dim0>=K || dim1 >=K)&&(dim0%DS == 0 && dim1%DS == 0) ){

        idx = samplesPerXRow*(dim1/DS) + dim0/DS - numRepeatedSamples;
        printf("Offset: %d, val: %lf, %lf \n", offset,result[numSamplesPerPlane*dim2 + K*K + idx ].x,result[numSamplesPerPlane*dim2 + K*K + idx ].y);
      }
      else{
        printf("Offset: %d, val: unsampled \n", offset);

      }
    }

  }

  if(dim2>=K){

    if(dim0%DS == 0 && dim1%DS == 0){
      idx = samplesPerXRow*(dim1/DS) + dim0/DS;
      printf("Offset: %d, val: %lf, %lf \n", offset,result[numSamplesPerPlane*K + samplesPerXRow*samplesPerXRow*(K-dim2) + idx ].x,result[numSamplesPerPlane*K + samplesPerXRow*samplesPerXRow*(K-dim2) + idx ].y);

    }
    else{

      printf("Offset: %d, val: unsampled \n", offset);
    }
  }



  offset =offset + 1;
}


}//end of print function

/* Helper function performing Cufft */

cudaError_t minibatch_CuFFT(int argc, char **argv, cufftDoubleComplex* h_a, cufftDoubleComplex* result, cufftDoubleComplex* d_a, cufftDoubleComplex* d_result, cufftDoubleComplex* unsampled_result){
	cudaError_t cudaStatus;
  cufftResult cufftStatus;
	cufftHandle *plans;
	int offset;
	int b;
  int k;
//	long *input_address, *output_address;
	 cufftDoubleComplex* d_fw_stage0;
   cufftDoubleComplex* d_temp1;
   cufftDoubleComplex* d_temp2; //to hold temporary group of B pencils of length K from the slab
   cufftDoubleComplex* d_temp3;
   cufftDoubleComplex* d_fw_stage1;
   cufftDoubleComplex* d_inv_stage1;
   cufftDoubleComplex* d_inv_stage2;


   int strideIdx;



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

   cudaMalloc((void**)&d_fw_stage0, sizeof(cufftDoubleComplex)*(NX*NY*K));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    exit(-1);
  }

  cudaMalloc((void**)&d_temp1, sizeof(cufftDoubleComplex)*B*K);
if (cudaGetLastError() != cudaSuccess){
  fprintf(stderr, "Cuda error: Failed to allocate\n");
  exit(-1);
}
cudaMalloc((void**)&d_temp2, sizeof(cufftDoubleComplex)*B*NZ);
if (cudaGetLastError() != cudaSuccess){
fprintf(stderr, "Cuda error: Failed to allocate\n");
exit(-1);
}
cudaMalloc((void**)&d_temp3, sizeof(cufftDoubleComplex)*B*(K + (NZ-K)/DS));
if (cudaGetLastError() != cudaSuccess){
fprintf(stderr, "Cuda error: Failed to allocate\n");
exit(-1);
}



    cudaMalloc((void**)&d_fw_stage1, sizeof(cufftDoubleComplex)*B*NZ);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    exit(-1);
  }

    cudaMalloc((void**)&d_inv_stage1, sizeof(cufftDoubleComplex)*NX*NY*(K + ((NZ-K)/DS)));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    exit(-1);
  }
  cudaMalloc((void**)&d_inv_stage2, sizeof(cufftDoubleComplex)*NX*NY*(K + ((NZ-K)/DS)));
if (cudaGetLastError() != cudaSuccess){
  fprintf(stderr, "Cuda error: Failed to allocate\n");
  exit(-1);
}



  //copy data
	 cudaStatus = cudaMemcpy(d_a, h_a, sizeof(cufftDoubleComplex)*K*K*K, cudaMemcpyHostToDevice);
          if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "dev_in cudaMalloc failed!");
                 exit(-1);
        }



	cout<<"creating first stage fft plan"<<endl;
	if (cufftPlanMany((plans+0), rank, t_size, inembed0, istride, idist, onembed0, ostride, odist, CUFFT_Z2Z, batch)!=CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		goto Error;
	};

	//Set callback function to do zero padding. Load callback function and attach it to plan_0
	cufftCallbackLoadZ h_pad_stage0;
  cudaStatus = cudaMemcpyFromSymbol(&h_pad_stage0, d_pad_stage0, sizeof(h_pad_stage0));
  if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "cudamemcpyfromsymbol failed!");
                 exit(-1);
        }


	cufftStatus = cufftXtSetCallback(*(plans+0), (void**)&h_pad_stage0, CUFFT_CB_LD_COMPLEX_DOUBLE, NULL);
  cout<< "cufftstatus:" << cufftStatus <<endl;

	cout<<"creating second stage plan"<<endl;

	//Create second plan that computes a batch of B pencils in each execution
  if (cufftPlanMany((plans+1), 1, &n_1,
                        &inembed1, B, 1, // *inembed, istride, idist
                        &onembed1, B, 1, // *onembed, ostride, odist
                        CUFFT_Z2Z, B) != CUFFT_SUCCESS){
                     fprintf(stderr, "CUFFT error: Plan creation failed");
                     goto Error;
                }

	//set callback functions for plan_1.
  //Load callback: Padding in Z dimension
  //Store callback: performing pointwise multiplication

cufftCallbackLoadZ h_pad_stage1;
cudaMemcpyFromSymbol(&h_pad_stage1, d_pad_stage1, sizeof(h_pad_stage1));
cufftXtSetCallback(*(plans+1),(void **)&h_pad_stage1,CUFFT_CB_LD_COMPLEX_DOUBLE,(void**)&d_offset);

cufftCallbackStoreZ h_greens_pointwise;
cudaMemcpyFromSymbol(&h_greens_pointwise, d_greens_pointwise, sizeof(h_greens_pointwise));
cufftXtSetCallback(*(plans+1),(void **)&h_greens_pointwise,CUFFT_CB_ST_COMPLEX_DOUBLE,(void**)&d_offset);
cudaDeviceSynchronize();

//invert the batch of B pencils.

if (cufftPlanMany((plans+2), 1, &n_2,
&inembed2, B, 1, // *inembed, istride, idist
&onembed2, B, 1, // *onembed, ostride, odist
CUFFT_Z2Z, B) != CUFFT_SUCCESS){
  fprintf(stderr, "CUFFT error: Plan creation failed");
  goto Error;
}

//attach callback. Samples will be set to zero. Output will still be B*NZ
cufftCallbackStoreZ h_sample_stage0;
cudaMemcpyFromSymbol(&h_sample_stage0, d_sample_stage0, sizeof(h_sample_stage0));
cufftXtSetCallback(*(plans+2),(void **)&h_sample_stage0,CUFFT_CB_ST_COMPLEX_DOUBLE,NULL);
cudaDeviceSynchronize();


// perform inverse transform in X and Y and sample
//
if (cufftPlanMany((plans+3), 2, n_3,
inembed3, 1, NX*NY, // *inembed, istride, idist
onembed3, 1,NX*NY,  // *onembed, ostride, odist|Previously ostride = 1, odist =NX*NY; We store the output as B pencils of size N
CUFFT_Z2Z, (K + ((NZ-K)/DS) )) != CUFFT_SUCCESS){
  fprintf(stderr, "CUFFT error: Plan creation failed");
  goto Error;
}

//
//attach callback. Samples will be set to zero. Output will still be NX*NY*NZ
cufftCallbackStoreZ h_sample_stage1;
cudaMemcpyFromSymbol(&h_sample_stage1, d_sample_stage1, sizeof(h_sample_stage1));
cufftXtSetCallback(*(plans+3),(void **)&h_sample_stage1,CUFFT_CB_ST_COMPLEX_DOUBLE,(void**)&d_result);
cudaDeviceSynchronize();

//-------------------------  execute plans ---------------------------//
cout<<"executing first stage fft"<<endl;


	 if (cufftExecZ2Z(*(plans+0), d_a, d_fw_stage0, CUFFT_FORWARD) != CUFFT_SUCCESS){
                        fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
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
        cudaStatus = cudaMemcpy(d_temp1+strideIdx*B, d_fw_stage0+offset+strideIdx*NX*NY, sizeof(cufftDoubleComplex)*B, cudaMemcpyDeviceToDevice);
        if (cudaStatus != cudaSuccess) {
               fprintf(stderr, "Pencils cudaMalloc failed!");
               exit(-1);
             }
      }




    	if (cufftExecZ2Z(*(plans+1), d_temp1, d_temp2, CUFFT_FORWARD) != CUFFT_SUCCESS){
    			fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
    			goto Error;
    		}


    	cudaStatus = cudaDeviceSynchronize();
      if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                    goto Error;
            }


       //Perform ifft
       if (cufftExecZ2Z(*(plans+2), d_temp2, d_temp3, CUFFT_INVERSE) != CUFFT_SUCCESS){
         fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
         goto Error;
       }

       //put the result in the appropriate place
       for(strideIdx =0; strideIdx< (K + ((NZ-K)/DS)); strideIdx++){
         cudaStatus = cudaMemcpy(d_inv_stage1+offset+strideIdx*NX*NY, d_temp3+strideIdx*B, sizeof(cufftDoubleComplex)*B, cudaMemcpyDeviceToDevice);
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
//do in place transform
//pass pointer to d_result in thru callback and write output there
   if (cufftExecZ2Z(*(plans+3), d_inv_stage1, d_inv_stage2, CUFFT_INVERSE) != CUFFT_SUCCESS){
     fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
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
  num_samples =   (K*K + (NX*NY - K*K)/(DS*DS))*K + ((NX*NY)/(DS*DS))*(NZ-K)/DS;
  cudaStatus = cudaMemcpy(result, d_result, sizeof(cufftDoubleComplex)*(num_samples), cudaMemcpyDeviceToHost);
 	//cudaStatus = cudaMemcpy(result, d_inv_stage1, sizeof(cufftDoubleComplex)*(num_samples), cudaMemcpyDeviceToHost);
 	if (cudaStatus != cudaSuccess){
 		fprintf(stderr, "cudaMemcpy failed!");
 		goto Error;

  }



  num_samples =   NX*NY*((NZ-K)/DS);
  cudaStatus = cudaMemcpy(unsampled_result, d_inv_stage2, sizeof(cufftDoubleComplex)*(num_samples), cudaMemcpyDeviceToHost);
 	//cudaStatus = cudaMemcpy(result, d_inv_stage1, sizeof(cufftDoubleComplex)*(num_samples), cudaMemcpyDeviceToHost);
 	if (cudaStatus != cudaSuccess){
 		fprintf(stderr, "cudaMemcpy failed!");
 		goto Error;

  }

  //print K-th plane. Use starting count = NX*NY*(K-1)
  // k=4;
  // count = NX*NY*(k-1);
  // while(count<NX*NY*k){
  //    cout<< count << ":" << result[count].x << ", " << result[count].y << endl;
  //    count = count + 1;
  //  }
  //print_XYplanes(result,NX,NY, (K+(NZ-K)/DS));


 	cudaStatus = cudaDeviceSynchronize();
 	if (cudaStatus != cudaSuccess) {
 		fprintf(stderr, "cudaDeviceSynchronize returned error code %d\n", cudaStatus);
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
  cudaFree(d_inv_stage2);
	cudaFree(d_result);
	cudaFree(d_offset);
	return cudaStatus;

}




//fftw functions

extern "C" void create_3Dplan_forward(fftw_plan *plan3d, double *temp, double *tempio, int m, int n, int k) {
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
  if(*(plan3d + 0)==NULL || *(plan3d + 1)==NULL)
  printf("NULL PLAN");

}
extern "C" void create_3Dplan_inverse(fftw_plan *plan3dinv, double *temp, double *tempio, int m, int n, int k) {
  // full 3D plan
  fftw_iodim s0[2], s1, d0, d1;



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

  *(plan3dinv + 0) = fftw_plan_guru_dft(2, s0, 1, &d0, (fftw_complex *) tempio, (fftw_complex *) temp, FFTW_BACKWARD, FFTW_MEASURE); //1 D fft
  *(plan3dinv + 1) = fftw_plan_guru_dft(1, &s1, 1, &d1, (fftw_complex*) temp, (fftw_complex*) temp, FFTW_BACKWARD, FFTW_MEASURE); // then 2d FFT
  if(*(plan3dinv+0)==NULL || *(plan3dinv + 1) ==NULL)
  printf("NULL PLAN");

}


extern "C" void execute_fftw_3d_plan_forward(fftw_plan *plan3D, double *tempio0, double *temp) {


  fftw_execute_dft(*(plan3D + 0), (fftw_complex*)tempio0, (fftw_complex*) temp);
  fftw_execute_dft(*(plan3D + 1), (fftw_complex*) temp, (fftw_complex*) temp);

}
extern "C" void execute_fftw_3d_plan_inverse(fftw_plan *plan3Dinv, double *tempio0, double *temp) {

  fftw_execute_dft(*(plan3Dinv + 0), (fftw_complex*)tempio0, (fftw_complex*) temp);
  fftw_execute_dft(*(plan3Dinv + 1), (fftw_complex*) temp, (fftw_complex*) temp);

/*

   i= 2*NX*NY*3;
  printf("FFTW output (first XY plane)\n");
  while(i<2*NX*NY*4){
     printf("%d:, %lf, %lf\n", i/2, *(temp+ i), *(temp+i+1));
     i= i + 2;
  }
*/

}




extern "C" int verify_with_fftw(double *fftw_output, double *cufft_output){

  int correct = 1;
  int i ;
  for( i = 0; i != 2*(NX * NY * NZ); ++i) {
    printf("Index: %d, FFTW: %lf, CUFFT: %lf\n",i, *(fftw_output + i), *(cufft_output + i));
        if((fabs(*(fftw_output + i) - *(cufft_output + i)) > 1e-3) || (isnan(*(fftw_output + i))) || (isnan(*(cufft_output + i)))) {
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
	cufftDoubleComplex *data;
	cufftDoubleComplex *small_cube;
	cufftDoubleComplex *result;
  double *fftw_input = new double[2*NX*NY*NZ];
  double *fftw_output = new double[2*NX*NY*NZ];
	double  *cufft_output = new double[2*NX*NY*NZ];
  int count,count1,count2;
  fftw_plan plan3d[2];
  fftw_plan plan3dinv[2];

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



        fftw_destroy_plan(*plan3d);
        fftw_destroy_plan(*plan3dinv);


        delete [] data;
        delete [] result;
        delete [] fftw_input;
        delete [] fftw_output;
        delete [] small_cube;
        return 0;
}
