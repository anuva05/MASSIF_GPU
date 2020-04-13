#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

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
  cufftDoubleComplex r;
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
