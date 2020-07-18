//FFTW part of the 3d code

// includes, system

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <float.h>
#define NX 4
#define NY 4
#define NZ 4


void create_3Dplan(fftw_plan *plan3d, double *temp, double *tempio, int m, int n, int k) {
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


void execute_fftw_3d_plan(fftw_plan *plan3D, double *tempio0, double *temp) {

 int i;
 

 printf("input\n");
 for(i=0;i!=(2*NX*NY*NZ);++i)
	printf("%lf\n",*(tempio0+i));


 printf("exe");
 fftw_execute_dft(*(plan3D + 0), (fftw_complex*)tempio0, (fftw_complex*) temp);

 printf("output\n");
 for( i = 0; i != 2*(NX * NY * NZ); ++i) {
    printf("%lf\n", *(temp+ i));
 }

 fftw_execute_dft(*(plan3D + 1), (fftw_complex*) temp, (fftw_complex*) temp);
 printf("FFTW output\n");
 for( i = 0; i != 2*(NX * NY * NZ);i++) {
    printf("%lf\n", *(temp+ i)); 
 }
printf("done");

	 }


int verify_with_fftw(double *fftw_output, double *cufft_output){

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



	double *fftw_input; 
	double *fftw_output;
	int count;
	int correct;
	fftw_plan plan3d[2];


	fftw_input= (double *)malloc(sizeof(double)*2*NX*NY*NZ);

	fftw_output= (double *)malloc(sizeof(double)*2*NX*NY*NZ);
	int i,j,k;
	//fftw_input = (std::complex<double> *) fftw_malloc( sizeof(std::complex<double>) *NX*NY

	//copy  to cpu

	//Creat FFTW plan on CPU
        printf("creating fftw plan\n");
        create_3Dplan(plan3d, fftw_output, fftw_input, NX, NY, NZ);

        //input for fftw on cpu..convert data from double to fftw_complex format
       
        count = 0;
        for(i=0;i<NZ;i++){
         for(j=0;j<NY;j++){
          for(k=0;k<NX;k++){
            fftw_input[count]= 0.3;
            fftw_input[count+1] = 0.0;
            count=count+2;
          }}}
        printf("FFT input initialized\n");


	//init output to zeros
	count = 0;
        for(i=0;i<NZ;i++){
         for(j=0;j<NY;j++){
          for( k=0;k<NX;k++){
            fftw_output[count]= 0.0;
            fftw_output[count+1] = 0.0;
            count=count+2;
          }}}



	//execute fftw 
	printf("executing FFTW plan");
        execute_fftw_3d_plan(plan3d, fftw_input, fftw_output);

/*

	for( i = 0; i != 2*(NX * NY * NZ); ++i) {
  	  printf("%lf\n", *(fftw_output+ i));   
	 }
*/
	printf("finished printing ouput");

	fftw_free(fftw_input);
	fftw_free(fftw_output);
	fftw_destroy_plan(*plan3d);
	//fftw_destroy_plan(*(plan3d+0));
        //fftw_destroy_plan(*(plan3d+1));
	return 0;
}
