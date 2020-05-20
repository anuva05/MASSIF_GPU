#include <iostream>
#include <string.h>
#include <math.h>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include "inputs.h"
#include "octree_table.cu"
/*******************************

 Main function

*********************************/





int main(int argc, char **argv){
	int blocks;
  int *octreeTable;
  int XB, YB, ZB;
  int *ds_rates;
	printf("kernel running\n");
	int numEntries; //cumulative Entries
	int xb, yb, zb, b;
	int current_ds;
  int dim0, dim1, dim2, sample_loc;
  int adjustedX, adjustedY, adjustedZ;
  XB = NX/OCTREE_FINEST;
  YB = NY/OCTREE_FINEST;
  ZB = NZ/OCTREE_FINEST;
  blocks = XB*YB*ZB;


 	ds_rates = (int*)malloc(sizeof(int)*blocks);
	octreeTable = (int*)malloc(sizeof(int)*blocks*5);
 /* Doing this on GPU
  cudaMalloc((void**)&ds_rates, sizeof(int)*blocks);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate octree table \n");
    exit(-1);
  }

	  cudaMalloc((void**)&octreeTable, sizeof(int)*blocks*5);
	  if (cudaGetLastError() != cudaSuccess){
	    fprintf(stderr, "Cuda error: Failed to allocate octree table \n");
	    exit(-1);
	  }
		//see if table is getting constructed
	  octree_table_construct<<<10,10>>>(ds_rates, octreeTable);
	  //
		*/



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





	  /* For eg first cube will be:
	  octreeTable[0]= 0; //x
	  octreeTable[1]= 0; //y
	  octreeTable[2]= 0; //z
	  octreeTable[3]= 1; //sample domain fully
	  octreeTable[4]= 0; //start idx in output array
	*/

	  numEntries=0;
	  b = 0;
	    for(zb=0; zb<ZB; zb++){
	      for(yb=0; yb<YB; yb++){
	        for(xb=0; xb<XB; xb++){

	          current_ds = ds_rates[b];
	          octreeTable[b*5]   = OCTREE_FINEST*xb;
	          octreeTable[b*5+1] = OCTREE_FINEST*yb;
	          octreeTable[b*5+2] = OCTREE_FINEST*zb;
	          octreeTable[b*5+3] = current_ds;
	          //size of the block is implicit: OCTREE_FINEST x OCTREE_FINEST x OCTREE x FINEST
						numEntries=numEntries+ (OCTREE_FINEST*OCTREE_FINEST*OCTREE_FINEST)/(current_ds*current_ds*current_ds);
	          octreeTable[b*5+4] = numEntries;//counts cumulative number of elements so far.


	          printf("STARTS: %d,%d,%d \t DS= %d \t numEntries = %d\n", octreeTable[b*5],octreeTable[b*5+1],octreeTable[b*5+2],octreeTable[b*5+3],octreeTable[b*5+4] );
	          b= b+1;
	  }}}


   // the following code is testing if sample location is correct given a certain cube and sampling rate 
		current_ds=2;
		for(dim2=Kprime;dim2<NZ;dim2++){
				for(dim1=Kprime;dim1<NY;dim1++){
					for(dim0=Kprime-K;dim0<Kprime;dim0++){
				  adjustedX = dim0 % OCTREE_FINEST;
				  adjustedY = dim1 % OCTREE_FINEST;
				  adjustedZ = dim2 % OCTREE_FINEST;


				if((adjustedX%current_ds==0)&&(adjustedY%current_ds==0)&&(adjustedZ%current_ds==0)){
		    sample_loc = (adjustedZ/current_ds)*(OCTREE_FINEST*OCTREE_FINEST)/(current_ds*current_ds) + (adjustedY/current_ds)*(OCTREE_FINEST/current_ds) + adjustedX/current_ds;
				printf("xyz:%d %d %d \t sample loc = %d\n", dim0, dim1, dim2, sample_loc );
			}
		 }}}


}
