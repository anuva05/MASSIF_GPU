//single precision
// for the first time, integrating the whole pipeline of forward fft, conv and inverse fft
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
#define B 32 // Number of pencils in one batch
#define NRANK 3
# define DS 4 //downsample rate
//need to define GPU constants for s0, c0

/* Callback functions for padding. Callbacks are element-wise */


__global__  void set_offset(int *d_offset, long *input_address, long *output_address){

printf("input add :%u\n", input_address);
//__global__ void set_offset(){
input_address= input_address + (int)*d_offset;
output_address=  output_address + *d_offset;
}


__device__ cufftComplex pad_stage0(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {

   // K x K x K is padded to NX x NY x K
    cufftComplex *input = (cufftComplex*) dataIn;
    int dim0 = ((offset) % NX) - 0;
    int dim1 = ((offset / NX) % NY) - 0;
    int dim2 = ((offset / (NX*NY)) ) - 0;

    //printf("padstage0 dims= %d,%d,%d\n", dim0, dim1, dim2);
     cufftComplex r;

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
      return *(input + offset);
   }
  else {

       r.x = 0.0;
       r.y = 0.0;
       return r;
   }
}
__device__ cufftCallbackLoadC d_pad_stage1 = pad_stage1;

//store callback
__device__ void greens_pointwise (void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr) {

    cufftComplex r;
    float green;
    /*// parameters for greens functions
    int delt[] = {1,1,1};
    int kx, ky, kz;
    int i,j,k,l;
    int xk[] = {0,0,0};
    float xknorm;
    int *d_offset = (int *) callerInfo;
    int start_pt;
    float *a; //3x3 matrix..linearized?
    start_pt = *d_offset;

    int dim0 = ((start_pt+ offset) % NX) - 0;
    int dim1 = (((start_pt + offset)/ NX) % NY) - 0;
    int dim2 = (((start_pt + offset) / (NX*NY)) ) - 0;

    //Compute green's function at this point
    green = 2;
    /*
    if (dim0<=NX/2){
      kx = dim0 - 1;
    }
    if(dim0>NX/2){
      kx = dim0 - NX-1;
    }
    if (dim1<=NY/2){
      ky = dim1 - 1;
    }
    if(dim1>NY/2){
      ky = dim1 - NY-1;
    }
    if (dim2<=NZ/2){
      kz = dim2 - 1;
    }
    if(dim2>NY/2){
      kz = dim2 - NZ-1;
    }
    xk[0]=kx/(delt[0]*NX);
    xk[1]=ky/(delt[1]*NY);
    xk[2]=kz/(delt[2]*NZ);
    xknorm = sqrt(xk[0]*xk[0] + xk[1]*xk[1] + xk[2]*xk[2]);

    if(xknorm!=0){
      xk[0] = xk[0]/xknorm;
      xk[1] = xk[1]/xknorm;
      xk[2] = xk[2]/xknorm;

          }

    if((dim0==NX/2 + 1)||(dim1==NY/2 + 1)||(dim2==NZ/2 + 1 )){
      //GAMMA FFT == -S0
      green = -s0; //taking only one component
    }
    else{
      for(i=0;i<3;i++){
        for(k=0;k<3;k++){
          a[i*3+k]=0;
          for(j=0;j<3;j++){
            for(l=0;l<3;l++){
              a[i*3 + k] = a[i*3+k] + c0[27*i + 9*k + 3*j + l]*xk[j]*xk[l];
            }

          }
        }
      }
    }

    a = pinv(a); //does this function exist?

   i=1;
   j=1;
   k=1;
   l=1;
   green= -a[3*i + j]*xk[k]*xk[l];

   //or, to compute full greens tensor in fourier domain, use the following:
    for(i=0;i<3;i++){
      for(k=0;k<3;k++){
        for(j=0;j<3;j++){
          for(l=0;l<3;l++){
            green[i,k,j,l]= -a[3*i + j]*xk[k]*xk[l];
          }}}}
*/

         //pointwise multiply
    r.x= green*element.x;
    r.y= 0;

	  ((cufftComplex*)dataOut)[offset] = r;
}

//to load the callback function onto the CPU, since it is a device function and resides only on GPU
__device__ cufftCallbackStoreC d_greens_pointwise = greens_pointwise;



__device__ void sample_stage0(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr) {

//if 'z' value is not in 0 to K or not one of the pre-specified planes, then ignore it.
// the output will be a set of XY planes stacked together with different z values
    int dim0, dim1, dim2;
    int downsample_rate = 4;
    dim0 = ((offset) % NX) - 0;
    dim1 = ((( offset)/ NX) % NY) - 0;
    dim2 = ((( offset) / (NX*NY)) ) - 0;


    if ((dim2>=0 && dim2 <K)||(dim2%downsample_rate == 0 )){
	     ((cufftComplex*)dataOut)[(1 * dim0) + (NX* dim1) + (NX*NY * dim2)] = element;
  }
}
__device__ cufftCallbackStoreC d_sample_stage0 = sample_stage0;



__device__ void sample_stage1(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr) {

// X and Y values should be corresponding to the samples we want to keep
    int dim0, dim1, dim2;
    int downsample_rate = 4;
    dim0 = ((offset) % NX) - 0;
    dim1 = ((( offset)/ NX) % NY) - 0;
    dim2 = ((( offset) / (NX*NY)) ) - 0;


    if (((dim0>=0 && dim0 <K)&&(dim1>=0 && dim1 <K))||(dim0%downsample_rate == 0 && dim1%downsample_rate == 0)){
	     ((cufftComplex*)dataOut)[(1 * dim0) + (NX* dim1) + (NX*NY * dim2)] = element;
  }
}
__device__ cufftCallbackStoreC d_sample_stage1 = sample_stage1;





/* Helper function performing Cufft */

cudaError_t minibatch_CuFFT(int argc, char **argv, cufftComplex* h_a, cufftComplex* result, cufftComplex* d_a, cufftComplex* d_output){
	cudaError_t cudaStatus;
  cufftResult cufftStatus;
	cufftHandle *plans;
	int count, offset;
	int b;
//	long *input_address, *output_address;
	 cufftComplex* d_fw_stage0;
   cufftComplex* d_temp; //to hold temporary group of B pencils of length K from the slab
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
   int onembed2 =  NX*NY*(K+(NZ-K)/DS); //
   int n_2 =NZ;
    //for fourth plan/*
   int inembed3[] =  {NX,NY};/// NX*NY*K;
   int onembed3[] =  {(K+(NX-K)/DS), (K+(NY-K)/DS)}; //??
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

  cudaMalloc((void**)&d_temp, sizeof(cufftComplex)*B*K);
if (cudaGetLastError() != cudaSuccess){
  fprintf(stderr, "Cuda error: Failed to allocate\n");
  exit(-1);
}

    cudaMalloc((void**)&d_fw_stage1, sizeof(cufftComplex)*B*NZ);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    exit(-1);
  }

    cudaMalloc((void**)&d_inv_stage1, sizeof(cufftComplex)*NX*NY*(K + (NZ-K)/DS));
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

	//Create second plan that computes a batch of B pencils in each execution. number of iterations required =N^2/B
  if (cufftPlanMany((plans+1), 1, &n_1,
                        &inembed1, B, 1, // *inembed, istride, idist
                        &onembed1, B, 1, // *onembed, ostride, odist|Previously ostride = 1, odist =NX*NY; We store the output as B pencils of size N
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

//invert the batch of B pencils. Need to figure out parameters, input, output sizes
if (cufftPlanMany((plans+2), 1, &n_2,
&inembed2, B, 1, // *inembed, istride, idist
&onembed2, NX*NY, 1, // *onembed, ostride, odist
CUFFT_C2C, B) != CUFFT_SUCCESS){
  fprintf(stderr, "CUFFT error: Plan creation failed");
  goto Error;
}
//attach callback. Size of output pencil = K + (NZ-K)/ds. Size of output array reqd = NX*NZ*(K + (NZ-K)/ds)
cufftCallbackStoreC h_sample_stage0;
cudaMemcpyFromSymbol(&h_sample_stage0, d_sample_stage0, sizeof(h_sample_stage0));
cufftXtSetCallback(*(plans+2),(void **)&h_sample_stage0,CUFFT_CB_ST_COMPLEX,NULL);
cudaDeviceSynchronize();


// perform inverse transform in X and Y and sample
if (cufftPlanMany((plans+3), 2, n_3,
inembed3, NX*NY, 1, // *inembed, istride, idist
onembed3, (K+ (NX-K)/DS)*(K+ (NY-K)/DS), 1, // *onembed, ostride, odist|Previously ostride = 1, odist =NX*NY; We store the output as B pencils of size N
CUFFT_C2C, K+(NZ-K)/DS) != CUFFT_SUCCESS){
  fprintf(stderr, "CUFFT error: Plan creation failed");
  goto Error;
}
//attach callback. Output size will be K*K*K + samples
cufftCallbackStoreC h_sample_stage1;
cudaMemcpyFromSymbol(&h_sample_stage1, d_sample_stage1, sizeof(h_sample_stage1));
cufftXtSetCallback(*(plans+3),(void **)&h_sample_stage1,CUFFT_CB_ST_COMPLEX,NULL);
cudaDeviceSynchronize();


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

    //the device variable being copied to should be a pointer..and pas address of host side int
		cudaStatus = cudaMemcpy(d_offset, &offset, sizeof(int), cudaMemcpyHostToDevice);
          if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "d_offset cudaMalloc failed!");
                 exit(-1);
        }

   cudaDeviceSynchronize();

  //copy B x K group into d_temp
  //copy B x K group into d_temp
  //ctr =0
  for(strideIdx =0; strideIdx< K; strideIdx++){

  for(i=0;i<B;i++){
    //count = offset + strideIdx*(NX*NY) + i;
    cudaMemcpy(d_temp+strideIdx*B, d_fw_stage0+offset+strideIdx*NX*NY, sizeof(cufftComplex)*B, cudaMemcpyDeviceToDevice);
    //printf("%d\n", count );
    //d_temp[ctr].x = d_fw_stage0[count].x;
    //d_temp[ctr].y = d_fw_stage0[count].y;
    //ctr = ctr + 1;
  }}
  



	if (cufftExecC2C(*(plans+1), d_temp, d_fw_stage1, CUFFT_FORWARD) != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
			goto Error;
		}


	cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                goto Error;
        }


//   //Perform ifft and sampling for these pencils
   if (cufftExecC2C(*(plans+2), d_fw_stage1, d_inv_stage1 + offset, CUFFT_INVERSE) != CUFFT_SUCCESS){
     fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
     goto Error;
   }
//
  cout<< "Offset:" << offset << endl;
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
               fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
               goto Error;
       }


	}//all batches processed

// //  Last stage, 2D inverse transform with sampling
   if (cufftExecC2C(*(plans+3), d_inv_stage1, d_output, CUFFT_INVERSE) != CUFFT_SUCCESS){
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
    num_samples =  ( K + (NX-K)/DS )*(K + (NY-K)/DS)*(K + (NZ-K)/DS);
 	cudaStatus = cudaMemcpy(result, d_output, sizeof(cufftComplex)*(num_samples), cudaMemcpyDeviceToHost);
 	if (cudaStatus != cudaSuccess ){
 		fprintf(stderr, "cudaMemcpy failed!");
 		goto Error;
 	cout<<"CUFFT output"<<endl;
  }
        cout<<"result" <<endl;
        count= 0;
 	while(count<num_samples){
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
	cudaFree(d_fw_stage0);
	cudaFree(d_fw_stage1);
	cudaFree(d_inv_stage1);
	cudaFree(d_output);
	cudaFree(d_offset);
	return cudaStatus;

}

//fftw functions

extern "C" void create_3Dplan(fftw_plan *plan3d, float *temp, float *tempio, int m, int n, int k) {
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


extern "C" void execute_fftw_3d_plan(fftw_plan *plan3D, float *tempio0, float *temp) {

 int i;
 fftw_execute_dft(*(plan3D + 0), (fftw_complex*)tempio0, (fftw_complex*) temp);
 fftw_execute_dft(*(plan3D + 1), (fftw_complex*) temp, (fftw_complex*) temp);
/*
  printf("FFTW output\n");
 for( i = 0; i != 2*(NX * NY * NZ); ++i) {
    printf("%lf\n", *(temp+ i));
 }*/
	 }


extern "C" int verify_with_fftw(float *fftw_output, float *cufft_output){

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










int main(int argc, char **argv){

  //Host variables
	cufftComplex *data;
	cufftComplex *small_cube;
	cufftComplex *result;
  float *fftw_input = new float[2*NX*NY*NZ];
  float *fftw_output = new float[2*NX*NY*NZ];
	float  *cufft_output = new float[2*NX*NY*NZ];
  int count;
  int correct;
  fftw_plan plan3d[2];

  //Device variables
  cufftComplex *d_result;//FULL N^3
  cufftComplex *d_a; //small cube K x K x K (technically real values)
  int final_samples;
  final_samples =  ( K + (NX-K)/DS )*(K + (NY-K)/DS)*(K + (NZ-K)/DS);


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
        data[NX*NY*i + NX*j + k ].x= i*j*k ;//i+j+k+0.3; //arbitrary value
        data[NX*NY*i + NX*j + k].y=0;

        small_cube[K*K*i + K*j + k].x = i*j*k; //i+j+k+0.3;//same value as data
        small_cube[K*K*i + K*j + k].y=0;

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

    //Create FFTW plan on CPU
    printf("creating fftw plan\n");
    create_3Dplan(plan3d, fftw_input, fftw_output, NX, NY, NZ);


   
    //input for fftw on cpu..convert data from float to fftw_complex format
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
    cout<<"executing FFTW plan and printing output"<<endl;
    execute_fftw_3d_plan(plan3d, fftw_input, fftw_output);


    //multiply by green's
   count = 0;
    for(int i=0;i<NZ;i++){
      for(int j=0;j<NY;j++){
        for(int k=0;k<NX;k++){
          fftw_output[count]= fftw_output[count]*2;   
	  fftw_output[count+1] = fftw_output[count+1]*2;
	  
          if(count<10)
	  cout<<fftw_output[count]<< ","<<fftw_output[count+1] <<endl;
          count=count+2;
        }}}


  

    delete [] data;
    delete [] result;
    return 0;
}
