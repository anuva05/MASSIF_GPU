#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <thrust/sort.h>
#include "skip_parallel.h"

__global__ void add(Skiplist *sl, OctreeNode a, int N)
{

    skiplist_insert(sl, a);
}

int main(void)
{
  OctreeNode *a = (OctreeNode*)malloc( sizeof(OctreeNode));
  OctreeNode *a_dev;
  int i,N;
  Skiplist *sl;
 

   N=1;
  // set heap size of 128 MB.
  CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024));
  size_t limit;
  cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
  printf("heap limit = %lu\n", limit);

  srand(time(NULL));
  CHECK(cudaMalloc(&a_dev, sizeof(OctreeNode)));

  a->val = 4;
  printf("done initializing\n");

  sl = skiplist_create();
  CHECK(cudaMemcpy(a_dev, a, sizeof(OctreeNode), cudaMemcpyHostToDevice));
/*
  add<<<100, 320>>>(sl, a_dev, N);
  CHECK(cudaDeviceSynchronize());
  printf("done inserting.\n");

  result = skiplist_gather(sl, &result_dim);
  result_sorted = (int *)malloc(result_dim * sizeof(int));
  memcpy(result_sorted, result, result_dim * sizeof(int));

  printf("done gathering.\n");
  printf("result_dim = %d\n", result_dim);

  thrust::sort(result_sorted, result_sorted + result_dim);
  printf("done sorting.\n");
  for (i = 0; i < result_dim; i++)
    if (result[i] != result_sorted[i])
      printf("mismatch at %d\n", i);
  printf("done checking.\n");

  free(a);
  free(result);
  free(result_sorted);
  cudaFree(a_dev);
  printf("starting skiplist_destroy...\n");
  skiplist_destroy(sl);
*/
  return 0;
}
