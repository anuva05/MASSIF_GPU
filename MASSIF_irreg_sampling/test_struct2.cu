#include <stdio.h>

class CudaInput
{
public:
int* octree;
double* result;

CudaInput(int final_samples) {
    octree = new int[10];

    result=  new double[final_samples];

    result[1]=10.0;
}
};

__global__ void useClass(CudaInput *cudaClass)
{   printf("i want to print\n" );
    printf("%lf\n", cudaClass->result[1]);
};




int main()
{
    CudaInput c(5);
    // create class storage on device and copy top level class
    CudaInput *d_c;

    cudaMalloc((void **)&d_c, sizeof(CudaInput));
    cudaMemcpy(d_c, &c, sizeof(CudaInput), cudaMemcpyHostToDevice);
    // make an allocated region on device for use by pointer in class
    int *temp_octree;
    double *temp_result;


    cudaMalloc((void **)&temp_octree, sizeof(int)*10);
    cudaMemcpy(temp_octree, c.octree, sizeof(int)*10, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&temp_result, sizeof(double)*5);
    cudaMemcpy(temp_result, c.result, sizeof(double)*5, cudaMemcpyHostToDevice);
    // copy pointer to allocated device storage to device class
    cudaMemcpy(&(d_c->octree), &temp_octree, sizeof(int *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_c->result), &temp_result, sizeof(double *), cudaMemcpyHostToDevice);
    useClass<<<1,1>>>(d_c);
    cudaDeviceSynchronize();
    return 0;
}
