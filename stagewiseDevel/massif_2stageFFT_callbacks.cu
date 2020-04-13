// Forward transform and multiplication with a constant (instead of green's function)
// Works for sizes upto 32x 32 x64 double precision. Doesnt work for 64x 64 x 64
//By: Anuva Kulkarni
// 10 March 2020
//Carnegie Mellon University

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
#define NX 32
#define NY 32
#define NZ 64
#define K 4  //dimension of small cube
#define B 256 // Number of pencils in one batch
#define NRANK 3


/* Callback functions for padding. Callbacks are element-wise */


__global__  void set_offset(int *d_offset, long *input_address, long *output_address){

printf("input add :%u\n", input_address);
//__global__ void set_offset(){
input_address= input_address + (int)*d_offset;
output_address=  output_address + *d_offset;
}


__device__ cufftDoubleComplex pad_stage0(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {

   // K x K x K is padded to NX x NY x K
    cufftDoubleComplex *input = (cufftDoubleComplex*) dataIn;
    int dim0 = ((offset) % NX) - 0;
    int dim1 = ((offset / NX) % NY) - 0;
    int dim2 = ((offset / (NX*NY)) ) - 0;

    //printf("padstage0 dims= %d,%d,%d\n", dim0, dim1, dim2);
     cufftDoubleComplex r;

   if (((0 <= dim0) && (dim0 < K)) && ((0 <= dim1) && (dim1 < K)) && ((0 <= dim2) && (dim2 < K))) {
     // printf("%lf\n", (*(input + ((1 * dim0) + (K * dim1) + (K*K * dim2)))).x);
      return *(input + ((1 * dim0) + (K* dim1) + (K*K * dim2)));
   }
  else {

       r.x = 0.0;
       r.y = 0.0;
       return r;
   }
}
__device__ cufftCallbackLoadZ d_pad_stage0 = pad_stage0;

//store callback
__device__ void greens_pointwise (void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo, void *sharedPtr) {

    cufftDoubleComplex r;
    double green;


    int *d_offset = (int *) callerInfo;
    int start_pt;
    start_pt = *d_offset;

   // printf("NX,NY,NZ= %d, %d, %d\n", NX,NY,NZ);
//printf("d_offset: %d, start_pt = %d\n", *d_offset, start_pt);


    int dim0 = ((start_pt+ offset) % NX) - 0;
    int dim1 = (((start_pt + offset)/ NX) % NY) - 0;
    int dim2 = (((start_pt + offset) / (NX*NY)) ) - 0;
    //int dim0 = (((start_pt + offset)/ 1) % NX) - 0;
    //int dim1 = (((start_pt + offset)/ NX) % NY) - 0;
    //int dim2 = (((start_pt + offset) / NX*NY) % NZ) - 0;

	//printf("dim0 = %d\n", dim0);
	// printf("dim1 = %d\n", dim1);
	// printf("dim2 = %d\n", dim2);
	//printf("net offset= %u, greens pointwise dims: %d, %d, %d \n", offset+start_pt, dim0, dim1, dim2);

//    if (((0 <= dim0) && (dim0 < K)) && ((0 <= dim1) && (dim1 < K)) && ((0 <= dim2) && (dim2 < K))) {


         //Compute green's function at this point
         green = 2;

         //pointwise multiply
        r.x= green*element.x;
        r.y= green*element.y;
	//printf("multiplied element: %lf, %lf\n", r.x,r.y);
//	printf("val = %lf, %lf, put here: %d\n",r.x,r.y, (1 * dim0) + (NX* dim1) + (NX*NY * dim2));


    dim0 = ((offset) % NX) - 0;
    dim1 = ((( offset)/ NX) % NY) - 0;
    dim2 = ((( offset) / (NX*NY)) ) - 0;
	  ((cufftDoubleComplex*)dataOut)[(1 * dim0) + (NX* dim1) + (NX*NY * dim2)] = r;

	//*(dataOut + (1 * dim0) + (NX* dim1) + (NX*NY * dim2)) = r;
  //  }
}

//to load the callback function onto the CPU, since it is a device function and resides only on GPU
__device__ cufftCallbackStoreZ d_greens_pointwise = greens_pointwise;

/* Helper function performing Cufft */

cudaError_t minibatch_CuFFT(int argc, char **argv, cufftDoubleComplex* h_a, cufftDoubleComplex* result, cufftDoubleComplex* d_a, cufftDoubleComplex* d_slab, cufftDoubleComplex* d_result){
	cudaError_t cudaStatus;
  cufftResult cufftStatus;
	cufftHandle *plans;
	int count, offset;
	int b;
	cufftDoubleComplex *slab;
//	long *input_address, *output_address;
	 cufftDoubleComplex* d_temp;
	 cufftDoubleComplex* t;

	t = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*B*NZ);

	slab = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(NX*NY*K));
	plans = (cufftHandle*) malloc(2 * sizeof(cufftHandle));


	  cudaMalloc((void**)&d_temp, sizeof(cufftDoubleComplex)*B*NZ);
	  if (cudaGetLastError() != cudaSuccess){
    	fprintf(stderr, "Cuda error: Failed to allocate\n");
 		 }


  // for the first plan
	 int rank = 2;
   int batch=K;
   int t_size[] = {NX,NY};
   int inembed[] = {NX,NY};
   int istride = 1;
   int idist= NX*NY;
   int odist = NX*NY;//how far next output signal is from current
   int onembed[] ={NX, NY};
   int ostride = 1;
   int *d_offset;
 //for second plan
   int inembed2 =  NX*NY*NZ;/// NX*NY*K;
   int onembed2 =  NX*NY*NZ; //??
   int n_2 =NZ;

	cudaMalloc((void**)&d_offset, sizeof(int));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
  }



/*
        cudaMalloc((void**)input_address, sizeof(long));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
  }


        cudaMalloc((void**)output_address, sizeof(long));
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
  }
*/

  //copy data
	 cudaStatus = cudaMemcpy(d_a, h_a, sizeof(cufftDoubleComplex)*K*K*K, cudaMemcpyHostToDevice);
          if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "dev_in cudaMalloc failed!");
                 exit(-1);
        }



	cout<<"creating first stage fft plan"<<endl;
	if (cufftPlanMany((plans+0), rank, t_size, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch)!=CUFFT_SUCCESS){
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

	//Create second plan that computes a batch of B pencils in each execution. number of iterations required =N^2/B
  if (cufftPlanMany((plans+1), 1, &n_2,
                        &inembed2, NX*NY, 1, // *inembed, istride, idist
                        &onembed2, NX*NY, 1, // *onembed, ostride, odist|Previously ostride = 1, odist =NX*NY; We store the output as B pencils of size N
                        CUFFT_Z2Z, B) != CUFFT_SUCCESS){
                     fprintf(stderr, "CUFFT error: Plan creation failed");
                     goto Error;
                }

	//set callback function for plan_1. Padding in Z dimension

  cufftCallbackStoreZ h_greens_pointwise;
  cudaMemcpyFromSymbol(&h_greens_pointwise, d_greens_pointwise, sizeof(h_greens_pointwise));
//  cufftXtSetCallback(*(plans+1),(void **)&h_greens_pointwise,CUFFT_CB_ST_COMPLEX_DOUBLE,0);


cufftXtSetCallback(*(plans+1),(void **)&h_greens_pointwise,CUFFT_CB_ST_COMPLEX_DOUBLE,(void**)&d_offset);
cudaDeviceSynchronize();
cout<<"executing first stage fft"<<endl;


	 if (cufftExecZ2Z(*(plans+0), d_a, d_slab, CUFFT_FORWARD) != CUFFT_SUCCESS){
                        fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
                        goto Error;
                }
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    goto Error;
  }



	//check output
	 cudaStatus = cudaMemcpy(slab, d_slab, sizeof(cufftDoubleComplex)*NX*NY*K, cudaMemcpyDeviceToHost);
          if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "cudaMemcpy deviceToHost failed!");
                 exit(-1);
        }


	cout<< "Intermediate slab FFT (2D)" << endl;
	  count =0 ;
    while(count<NX*NY*K){
              //  cout<< slab[count].x << endl;
                count++;
        }

	//next step - transform in Z using tiling and minibatches
	for(b=0;b<(NX*NY)/B ;b++){

		cout<<"Executing batch number"<< b << endl;
		///////// new method ///////////
		offset = (b*B); //each batch processes B pencils
		int arr[] = {offset};
              //the device variable being copied to should be a pointer..and pas address of host side int
		cudaStatus = cudaMemcpy(d_offset, &offset, sizeof(int), cudaMemcpyHostToDevice);
          if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "d_offset cudaMalloc failed!");
                 exit(-1);
        }
	
 cudaDeviceSynchronize();
		if (cufftExecZ2Z(*(plans+1), d_slab + offset, d_result + offset, CUFFT_FORWARD) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
			goto Error;
		}

//	cout<< "Offset:" << offset << endl;
	cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                goto Error;
        }


//	cudaStatus = cudaMemcpy(t, d_temp, sizeof(cufftDoubleComplex)*(B*NZ), cudaMemcpyDeviceToHost);

//	count= 0;
 //       while(count<B*NZ){
  //              cout<<t[count].x <<endl;
   //             ++count;}

	}//all batches processed


 //check output
	cudaStatus = cudaMemcpy(result, d_result, sizeof(cufftDoubleComplex)*(NX*NY*NZ), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess ){
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	cout<<"CUFFT output"<<endl;

 }
       cout<<"result" <<endl;
      /* count= 0;
	while(count<NX*NY*NZ){
		cout<<result[count].x << "," << result[count].y<<endl;
		++count;}
      */
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaGetLastError();

Error:
	//Delete the CUFFT Plan number 0
	cufftDestroy(*(plans + 0));
  //Delete the CUFFT Plan number 1
  cufftDestroy(*(plans + 1));
	free(plans);
	cudaFree(d_a);
	cudaFree(d_slab);
	cudaFree(d_result);
	cudaFree(d_offset);
	return cudaStatus;

}

//fftw functions

extern "C" void create_3Dplan(fftw_plan *plan3d, double *temp, double *tempio, int m, int n, int k) {
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
   if(*(plan3d+0)==NULL || *(plan3d + 1) ==NULL)
        printf("NULL PLAN");

}


extern "C" void execute_fftw_3d_plan(fftw_plan *plan3D, double *tempio0, double *temp) {

 int i;
 fftw_execute_dft(*(plan3D + 0), (fftw_complex*)tempio0, (fftw_complex*) temp);
 fftw_execute_dft(*(plan3D + 1), (fftw_complex*) temp, (fftw_complex*) temp);
/* printf("FFTW output\n");
 for( i = 0; i != 2*(NX * NY * NZ); ++i) {
    printf("%lf\n", *(temp+ i));
 }*/
	 }


extern "C" int verify_with_fftw(double *fftw_output, double *cufft_output){

  int correct = 1;
  int i ;
  for( i = 0; i != 2*(NX * NY * NZ); ++i) {
   // printf("Index: %d, FFTW: %lf, CUFFT: %lf\n",i, *(fftw_output + i), *(cufft_output + i));
        if((fabs(*(fftw_output + i) - *(cufft_output + i)) > 1e-3) || (isnan(*(fftw_output + i))) || (isnan(*(cufft_output + i)))) {
              correct = 0;
                  }
                    }

        printf("Correctness: %d\n", correct);
        return correct;
}










int main(int argc, char **argv){

  //Host variables
	cufftDoubleComplex *data;
	cufftDoubleComplex *small_cube;
	cufftDoubleComplex *result;
  double *fftw_input = new double[2*NX*NY*NZ];
  double *fftw_output = new double[2*NX*NY*NZ];
	double  *cufft_output = new double[2*NX*NY*NZ];
  int count;
  int correct;
  fftw_plan plan3d[2];

  //Device variables
  cufftDoubleComplex *d_result;//FULL N^3
  cufftDoubleComplex *d_a; //small cube K x K x K (technically real values)
  cufftDoubleComplex *d_slab; //first stage of transform on small cube



 	//allocating host side arrays
	result = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*(NX*NY*NZ));
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


  cudaMalloc((void **)&d_slab, sizeof(cufftDoubleComplex)*NX*NY*K);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return 0;
  }


  //**TEMPORARY** the output is going to materialize the full cube for simplicity
  cudaMalloc((void**)&d_result, sizeof(cufftDoubleComplex)*(NX*NY*NZ));
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
        data[NX*NY*i + NX*j + k ].x= i*j*k ;//i+j+k+0.3; //arbitrary value
        data[NX*NY*i + NX*j + k].y=0;

        small_cube[K*K*i + K*j + k].x = i*j*k; //i+j+k+0.3;//same value as data
        small_cube[K*K*i + K*j + k].y=0;

      }}}



  // Running cuFFT

  cout << "Run cufft" <<endl;
  cudaError_t cudaStatus = minibatch_CuFFT(argc, argv, small_cube, result, d_a, d_slab, d_result);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "CuFFT failed!");
    return 1;
  }


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




    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
    }

    //Create FFTW plan on CPU
    printf("creating fftw plan\n");
    create_3Dplan(plan3d, fftw_input, fftw_output, NX, NY, NZ);

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
    cout<<"executing FFTW plan"<<endl;
    execute_fftw_3d_plan(plan3d, fftw_input, fftw_output);


    //multiply by green's
   count = 0;
    for(int i=0;i<NZ;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NX;k++){
          fftw_output[count]= fftw_output[count]*2;
          fftw_output[count+1] = fftw_output[count+1]*2;
          count=count+2;
        }}}


    printf("checking correctness\n");
    verify_with_fftw(fftw_output, cufft_output);


    delete [] data;
    delete [] result;
    return 0;
}
