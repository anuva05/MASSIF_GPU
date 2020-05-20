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

      return *(input + ((1 * dim0) + (K* dim1) + (K*K * dim2)));
   }
  else {

       r.x = 0.0;
       r.y = 0.0;
       return r;
   }
}
__device__ cufftCallbackLoadZ d_pad_stage0 = pad_stage0;


__device__ cufftDoubleComplex pad_stage1(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {

   // B pencils of length K are padded to B pencils of length NZ.
   //Input layout has x has fastest varying dimension, followed by y, then z.
   //idist = 1, istride = B
   //hence, if offset < B*K, then return value. Else return 0
    cufftDoubleComplex *input = (cufftDoubleComplex*) dataIn;
    cufftDoubleComplex r;

   if (offset<B*K) {
      r =*(input + offset);
      return r;
   }
  else {

       r.x = 0.0;
       r.y = 0.0;
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


__device__ cufftCallbackStoreZ d_greens_pointwise = greens_pointwise;


/* --------- Callbacks that perform sampling -------------*/

__device__ void sample_stage0(void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo, void *sharedPtr) {


  cufftDoubleComplex r;
  int z;
  int el;
  z = offset/B;
  el = offset%B;
  if (z>=0 && z<K){
    ((cufftDoubleComplex*)dataOut)[offset] = element;
  }

  if ((z>=K )&&(z<Kprime)) {
    if (z%DS1 == 0){
      ((cufftDoubleComplex*)dataOut)[B*(K + (z/DS1)-1) + el] = element;
    }

  }
  if (z>=Kprime){
    if (z%DS2 == 0){
      ((cufftDoubleComplex*)dataOut)[B*(Kprime + (z/DS2)-1) + el] = element;
    }

  }


}

__device__ cufftCallbackStoreZ d_sample_stage0 = sample_stage0;



__device__ void sample_stage1(void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo, void *sharedPtr) {

  int dim0, dim1, dim2;
  int off = static_cast<int>(offset);
  int numSamplesPerPlane, samplesPerXRow, numRepeatedSamples;
  int idx;
  cufftDoubleComplex *d_result = (cufftDoubleComplex*)callerInfo;


  dim2 = ( off)/(NX*NY)  ;
  dim1 = (off - NX*NY*dim2)/NX;
  dim0 = (off - NX*NY*dim2 - NX*dim1);

  numSamplesPerPlane = K*K + (NX*NY - K*K)/(DS1*DS1);
  samplesPerXRow = NX/DS;
  numRepeatedSamples = (K*K)/(DS1*DS1);//in first K planes


  //Put the element in the same place for the cufft output array - no change here
  ((cufftDoubleComplex*)dataOut)[offset] = element;


  //create custom array to store samples

  //For points that are in the domain
  if(dim2<K){
    if((dim0>=0 && dim0 <K)&&(dim1>=0 && dim1 <K)){
      //store in appropriate location in the compressed output
      d_result[numSamplesPerPlane*dim2 + K*dim1 + dim0] =element;
    }

    if((dim0>=K || dim1 >=K)&&(dim0%DS1 == 0 && dim1%DS1 == 0) ){
      idx = samplesPerXRow*(dim1/DS1) + dim0/DS1 - numRepeatedSamples;
      d_result[numSamplesPerPlane*dim2 + K*K + idx ] =element;
    }

  }


  //For points that are not in the domain
  if(dim2>=K){

    if(dim0%DS1 == 0 && dim1%DS1 == 0){
      idx = samplesPerXRow*(dim1/DS1) + dim0/DS1;
      d_result[numSamplesPerPlane*K + samplesPerXRow*samplesPerXRow*(K-dim2) + idx ] =element;

    }
  }

  // startx = dim0 / finest
  //similarly starty ,startz
  //retrieve sampling rate and startindex in the new datastructure
  //store octree as, size is same for now: (start points) (sampling rate) (start index into outptu)
  //where to put sample : start_index + ((dim0)%8)/sampling rate
  //lets keep finest=  K (sz of domain)
  
}



__device__ cufftCallbackStoreZ d_sample_stage1 = sample_stage1;
