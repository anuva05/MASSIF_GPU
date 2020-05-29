//Integrating the whole pipeline of forward fft, convolution and inverse fft
//Remove the samples that we want to leave out, using case /* value */:allbacks
//This approach saves memory


#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <fftw3.h>
#include <float.h>
#include <fstream>

void write_to_csv(CudaInput *c){


        int XB, YB, ZB;
        int x,y,z;
        int adjustedX, adjustedY, adjustedZ;
        int xstart, ystart, zstart;
        int start_loc, sample_loc;
        int blocks, current_ds;
        int b, prevNumEntries,count;
        int numEntries;
        cufftDoubleComplex element;

       std::ofstream outfile;
       outfile.open ("samples_cufft.csv");
       outfile << "x,y,z,real_val, imag_val\n";
       //decode octree


         XB = NX/OCTREE_FINEST;
         YB = NY/OCTREE_FINEST;
         ZB = NZ/OCTREE_FINEST;
         blocks = XB*YB*ZB;

         b = 0;
         prevNumEntries = 0;
           while(b<blocks){

                 xstart= c->octree[b*5] ;
                 ystart = c->octree[b*5+1];
                 zstart = c->octree[b*5+2];
                 current_ds = c->octree[b*5+3];
                 numEntries= c->octree[b*5+4];

                 if( b > 0){
                   start_loc =  c->octree[(b-1)*5+4 ];}
                 else{
                   start_loc = 0;
                 }


                 count  = 0;


                 for (z=zstart; z< zstart + OCTREE_FINEST; z++){
                   for(y=ystart; y<ystart + OCTREE_FINEST;y++){
                     for(x=xstart; x<xstart+OCTREE_FINEST;x++){
                       adjustedX = x % OCTREE_FINEST;
                       adjustedY = y % OCTREE_FINEST;
                       adjustedZ = z % OCTREE_FINEST;
                       if((adjustedX%current_ds==0)&&(adjustedY%current_ds==0)&&(adjustedZ%current_ds==0)){
                        sample_loc = (adjustedZ/current_ds)*(OCTREE_FINEST*OCTREE_FINEST)/(current_ds*current_ds) + (adjustedY/current_ds)*(OCTREE_FINEST/current_ds) + adjustedX/current_ds;

                        element.x = c->result[start_loc + sample_loc].x;
                        element.y = c->result[start_loc + sample_loc].y;
                        count = count + 1;
                        //write to  csv
                        outfile << x <<"," << y << "," << z << "," << element.x << "," << element.y << "\n";

                       }
                     }
                   }
                 }

                if (count != numEntries - prevNumEntries){
                  printf("ERROR IN WRITE_TO_CSV\n" );
                }
                prevNumEntries = numEntries;
           //now go to next block
           b= b+1;

         }




       outfile.close();


}


void write_fftw_to_csv(double *arr){
  int count;

  std::ofstream outfile;
  outfile.open ("samples_fftw.csv");
  outfile << "x,y,z,real_val, imag_val\n";

  count = 0;
  for(int i=0;i<NZ;i++){
  	for(int j=0;j<NY;j++){
  		for(int k=0;k<NX;k++){

        outfile << k <<"," << j << "," << i << "," << arr[count] << "," << arr[count+1]<< "\n";

  			count=count+2;
  		}}}
  outfile.close();

}//write fftw to csv



//function to print
void printResult(cufftDoubleComplex *result, int samples){
int dim0, dim1, dim2;
int numSamplesPerPlane, samplesPerXRow, numRepeatedSamples;
int idx;
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

cudaError_t minibatch_CuFFT(int argc, char **argv, cufftDoubleComplex* h_a, cufftDoubleComplex* result, cufftDoubleComplex* d_a, cufftDoubleComplex* d_result, cufftDoubleComplex* unsampled_result, int *final_samples){
  cudaError_t cudaStatus;
  cufftResult cufftStatus;
  cufftHandle *plans;
  int offset;
  int b;

  //	long *input_address, *output_address;
  cufftDoubleComplex* d_fw_stage0;
  cufftDoubleComplex* d_temp1;
  cufftDoubleComplex* d_temp2; //to hold temporary group of B pencils of length K from the slab
  cufftDoubleComplex* d_temp3;
  cufftDoubleComplex* d_fw_stage1;
  cufftDoubleComplex* d_inv_stage1;
  cufftDoubleComplex* d_inv_stage2;


  int i,strideIdx;
  int blocks =(NX/OCTREE_FINEST)*(NY/OCTREE_FINEST)*(NZ/OCTREE_FINEST);



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
  int inembed2 =  B*NZ;
  int onembed2 =  B*NZ;
  int n_2 =NZ;
  //fourth plan
  int inembed3[] =   {NX,NY};
  int onembed3[] = {NX,NY};
  int n_3[] = {NX,NY};
  int num_samples;

  /// little experiment with the struct
  CudaInput c;

  printf("num entries : %d\n", c.numEntries);
  // create class storage on device and copy top level class
  CudaInput *d_c;

  cudaMalloc((void **)&d_c, sizeof(CudaInput));
  cudaMemcpy(d_c, &c, sizeof(CudaInput), cudaMemcpyHostToDevice);
  // make an allocated region on device for use by pointer in class
  int *temp_octree;
  int *temp_ds_rates;
  int *temp_numEntries;
  cufftDoubleComplex *temp_result;


  cudaMalloc((void **)&temp_octree, sizeof(int)*blocks*5);
  cudaMemcpy(temp_octree, c.octree, sizeof(int)*blocks*5, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&temp_ds_rates, sizeof(int)*blocks);
  cudaMemcpy(temp_ds_rates, c.ds_rates, sizeof(int)*blocks, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&temp_numEntries, sizeof(int));
  cudaMemcpy(temp_numEntries, &c.numEntries, sizeof(int), cudaMemcpyHostToDevice);

  //c.result =  new cufftDoubleComplex [*c.numEntries];
  printf("Init c.result to zeros..\n" );
  c.result = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex)*(c.numEntries));
  for(i=0;i< c.numEntries;i++){
    c.result[i].x = 0.0;
    c.result[i].y = 0.0;
  }
  cudaMalloc((void **)&temp_result, sizeof(cufftDoubleComplex)*(c.numEntries));
  cudaMemcpy(temp_result, c.result, sizeof(cufftDoubleComplex)*(c.numEntries), cudaMemcpyHostToDevice);

  // copy num entries too ?

  // copy pointer to allocated device storage to device class
  cudaMemcpy(&(d_c->octree), &temp_octree, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_c->ds_rates), &temp_ds_rates, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_c->numEntries), &temp_numEntries, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_c->result), &temp_result, sizeof(double *), cudaMemcpyHostToDevice);


  cudaMalloc((void**)&d_offset, sizeof(int));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate doffset\n");
  }

  cudaMalloc((void**)&d_fw_stage0, sizeof(cufftDoubleComplex)*(NX*NY*K));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate fw_stage0 \n");
    exit(-1);
  }

  cudaMalloc((void**)&d_temp1, sizeof(cufftDoubleComplex)*B*K);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate temp1 \n");
    exit(-1);
  }
  cudaMalloc((void**)&d_temp2, sizeof(cufftDoubleComplex)*B*NZ);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate temp2\n");
    exit(-1);
  }
  cudaMalloc((void**)&d_temp3, sizeof(cufftDoubleComplex)*B*(K + (NZ-K)/DS));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate temp3 \n");
    exit(-1);
  }

  cudaMalloc((void**)&d_fw_stage1, sizeof(cufftDoubleComplex)*B*NZ);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate fw_stage1 \n");
    exit(-1);
  }

  cudaMalloc((void**)&d_inv_stage1, sizeof(cufftDoubleComplex)*NX*NY*(K + ((NZ-K)/DS)));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate invstage1 \n");
    exit(-1);
  }
  cudaMalloc((void**)&d_inv_stage2, sizeof(cufftDoubleComplex)*NX*NY*(K + ((NZ-K)/DS)));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate inv stage2\n");
    exit(-1);
  }



  //copy data
  /*
   count = 0;
   while(count<K*K*K){
     printf("input = %lf \n", h_a[count].x );
     count = count + 1;
   }*/

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
if (cufftPlanMany((plans+3), 2, n_3,
inembed3, 1, NX*NY, // *inembed, istride, idist
onembed3, 1,NX*NY,  // *onembed, ostride, odist
CUFFT_Z2Z, (K + ((NZ-K)/DS) )) != CUFFT_SUCCESS){
  fprintf(stderr, "CUFFT error: Plan creation failed");
  goto Error;
}


//attach callback. Samples will be set to zero. Output will still be NX*NY*NZ
cufftCallbackStoreZ h_sample_stage1;
cudaMemcpyFromSymbol(&h_sample_stage1, d_sample_stage1, sizeof(h_sample_stage1));
//cufftXtSetCallback(*(plans+3),(void **)&h_sample_stage1,CUFFT_CB_ST_COMPLEX_DOUBLE,(void**)&d_result);
cufftXtSetCallback(*(plans+3),(void **)&h_sample_stage1,CUFFT_CB_ST_COMPLEX_DOUBLE,(void**)&d_c);
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

      //copy B x K group of pencils into d_temp1

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
                    fprintf(stderr, "cudaDeviceSynchronize1 returned error code %d after launching addKernel!\n", cudaStatus);
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
                   fprintf(stderr, "cudaDeviceSynchronize2 returned error code %d after launching addKernel!\n", cudaStatus);
                   goto Error;
           }


	}//all batches processed

//Last stage, 2D inverse transform with sampling
//In place transform
//pass pointer to d_result in thru callback and write output there
   if (cufftExecZ2Z(*(plans+3), d_inv_stage1, d_inv_stage2, CUFFT_INVERSE) != CUFFT_SUCCESS){
     fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
     goto Error;
   }
 cudaStatus = cudaDeviceSynchronize();
       if (cudaStatus != cudaSuccess) {
               fprintf(stderr, "cudaDeviceSynchronize3 returned error code %d after launching addKernel!\n", cudaStatus);
               goto Error;
       }

 // ------------------------ Finished executing cuffts ---------------------------//
//Copy out output

//// testing if there is data in the struct

  printf("about to copy out the value in struct\n" );
  c.numEntries= 100;
  cudaStatus = cudaMemcpy(&c.numEntries, temp_numEntries, sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess){
 		fprintf(stderr, "cudaMemcpy to c.numEntries failed!");
 		goto Error;

  }
  printf("copied numentries: %d\n", c.numEntries );


  cudaStatus =  cudaMemcpy( c.result, temp_result, sizeof(cufftDoubleComplex)*(c.numEntries), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess){
    fprintf(stderr, "cudaMemcpy to c.result failed!");
    goto Error;

  }
  printf("copied first entry to c_result: %lf\n", c.result[0].x );

  //copy the result into an output array
  i= 0;
  double sum ;
  sum = 0;
  while(i<c.numEntries){
    result[i].x = c.result[i].x;
    result[i].y = c.result[i].y;
    sum = sum + (result[i].x*result[i].x +result[i].y*result[i].y);
  //  if (i<20) {
    //  printf("sample number = %d,val= %lf +i*%lf\n",i, c.result[i].x , c.result[i].y);
  //  }
    i = i + 1;
  }
  printf("SUM OF SQUARES OF SAMPLES  = %lf\n",sum );

  //Write the decoded octree samples to a csv file
  write_to_csv(&c);

  num_samples =   NX*NY*((NZ-K)/DS);
  cudaStatus = cudaMemcpy(unsampled_result, d_inv_stage2, sizeof(cufftDoubleComplex)*(num_samples), cudaMemcpyDeviceToHost);
 	if (cudaStatus != cudaSuccess){
 		fprintf(stderr, "cudaMemcpy failed!");
 		goto Error;

  }

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
  cudaFree(d_c);
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
