#include<stdio.h>

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
/*
    int i;
  i= 2*NX*NY*3;
 printf("FFTW output (first XY plane)\n");
 while(i<2*NX*NY*4){
    printf("%d:, %lf , %lf\n", i/2, *(temp+ i), *(temp+i+1));
    i= i + 2;
  }*/

}
extern "C" void execute_fftw_3d_plan_inverse(fftw_plan *plan3Dinv, double *tempio0, double *temp) {

  fftw_execute_dft(*(plan3Dinv + 0), (fftw_complex*)tempio0, (fftw_complex*) temp);
  fftw_execute_dft(*(plan3Dinv + 1), (fftw_complex*) temp, (fftw_complex*) temp);
  fftw_cleanup_threads();


/*
      int i;
   i= 2*NX*NY*3;
  printf("FFTW output (first XY plane)\n");
  while(i<2*NX*NY*4){
     printf("%d:, %lf, %lf\n", i/2, *(temp+ i), *(temp+i+1));
     i= i + 2;
  }
*/

}

/*


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
*/
