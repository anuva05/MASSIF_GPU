#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>


class CudaInput
{
public:
int* octree;
int* ds_rates;
int numEntries;
cufftDoubleComplex * result;

CudaInput() {

    //this should be computed by another function
    //int *octree;
    //int *ds_rates;
    //int numEntries = 0;

    //temporary variables

    int blocks;
    int XB, YB, ZB;
    int xb, yb, zb, b;
    int current_ds;

    XB = NX/OCTREE_FINEST;
    YB = NY/OCTREE_FINEST;
    ZB = NZ/OCTREE_FINEST;
    blocks = XB*YB*ZB;

    //malloc for the main arrays
    ds_rates= new int [blocks];
    octree = new int [blocks*5];



    //hard code ds rates for now
    ds_rates[0]=1; //domain
    b=0;
    for(zb=0; zb<ZB; zb++){
      for(yb=0; yb<YB; yb++){
        for(xb=0; xb<XB; xb++){

          if (zb*OCTREE_FINEST >= Kprime){
            ds_rates[b] = DS2;
          }
          if ((zb*OCTREE_FINEST < Kprime)&&(zb*OCTREE_FINEST >= K)){
            ds_rates[b] = DS1;
          }
          if (zb*OCTREE_FINEST < K){
            ds_rates[b]=1;
          }
          b= b+1;

  }}}

/*
   b =0;
   while(b<blocks){
     printf("ds rate for %d= %d \n",b,ds_rates[b] );
     b=b+1;
   }
   */

    numEntries=0;//number of samples in the result
    b = 0;
      for(zb=0; zb<ZB; zb++){
        for(yb=0; yb<YB; yb++){
          for(xb=0; xb<XB; xb++){

            current_ds = ds_rates[b];
            octree[b*5]   = OCTREE_FINEST*xb;
            octree[b*5+1] = OCTREE_FINEST*yb;
            octree[b*5+2] = OCTREE_FINEST*zb;
            octree[b*5+3] = current_ds;
            //size of the block is implicit: OCTREE_FINEST x OCTREE_FINEST x OCTREE x FINEST
            numEntries= numEntries+ (OCTREE_FINEST*OCTREE_FINEST*OCTREE_FINEST)/(current_ds*current_ds*current_ds);
            octree[b*5+4] = numEntries;

            //printf("STARTS: %d,%d,%d   DS= %d    numEntries = %d\n", octree[b*5],octree[b*5+1],octree[b*5+2],octree[b*5+3],octree[b*5+4] );
            b= b+1;
    }}}


    //this should just be a pointer a cufftDoubleComplex array of size final_samples

    //cufftDoubleComplex*  result;
    int final_samples;
    final_samples=numEntries;

    printf("final samples = %d\n", final_samples);
    //result = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex)*final_samples);
    //result = new cufftDoubleComplex [final_samples];
    //printf("Finished constructor in octree struct. final samples = %d\n", final_samples );

}
//write a desctructor too
~CudaInput(){

  delete [] octree;
  delete [] ds_rates;
  delete [] result;

}
};//finished creating struct




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
  int idx;
  //cufftDoubleComplex *d_result = (cufftDoubleComplex*)callerInfo;

  CudaInput *in = (CudaInput *)callerInfo;

  /* Logic: get the block in which current (x,y,z) computed from offset belongs to
   This is given by block_number.
   Then get the numEntries total till the prev block. This is start_loc
   Then get current block's DS rate in current_ds
   Then get the sample loc
   Put the value at (x,y,z) in start_loc + sample_loc


  */


  int cubex, cubey, cubez;
  int block_number, current_ds;
  int start_loc, sample_loc;
  int adjustedX, adjustedY, adjustedZ;
  int XB, YB, ZB;
  XB = NX/OCTREE_FINEST;
  YB = NY/OCTREE_FINEST;
  ZB = NZ/OCTREE_FINEST;

  dim2 = ( off)/(NX*NY)  ;
  dim1 = (off - NX*NY*dim2)/NX;
  dim0 = (off - NX*NY*dim2 - NX*dim1);

  //make adjustment for sample_stage0
  //currently DS1==DS1
  if (dim2 >= K){

    dim2 = DS1*(dim2 - K ) + K;


  }


  cubex = dim0/ OCTREE_FINEST;
  cubey = dim1/ OCTREE_FINEST;
  cubez = dim2/ OCTREE_FINEST;

  block_number = cubex + XB*cubey + XB*YB*cubez;


  //printf("first cube vals = %d,%d,%d,%d, %d\n",  in->octree[0],in->octree[1],in->octree[2],in->octree[3], in->octree[4]);


  //start loc in output array=where prev block ends
  if( block_number > 0){
    start_loc =  in->octree[(block_number-1)*5+4 ];}
  else{
    start_loc = 0;
  }

  //printf("Computed vals: cubex, cubey, cubez = %d,%d,%d, block=%d\n", cubex, cubey, cubez, block_number );

  //sampling rate of this block
  current_ds = in->octree[block_number*5 + 3];

  //printf("Retrieved vals: Start loc: %d, current ds = %d \n",  start_loc, current_ds );

  adjustedX = dim0 % OCTREE_FINEST;
  adjustedY = dim1 % OCTREE_FINEST;
  adjustedZ = dim2 % OCTREE_FINEST;



  if((adjustedX%current_ds==0)&&(adjustedY%current_ds==0)&&(adjustedZ%current_ds==0)){
   sample_loc = (adjustedZ/current_ds)*(OCTREE_FINEST*OCTREE_FINEST)/(current_ds*current_ds) + (adjustedY/current_ds)*(OCTREE_FINEST/current_ds) + adjustedX/current_ds;
   //printf("Adjusted: %d, %d,%d \n",  adjustedX, adjustedY, adjustedZ);
   //printf("start loc = %d, sample loc = %d\n", start_loc, sample_loc);

   in->result[start_loc + sample_loc]=element;
  }


  ((cufftDoubleComplex*)dataOut)[offset] = element;


}



__device__ cufftCallbackStoreZ d_sample_stage1 = sample_stage1;
