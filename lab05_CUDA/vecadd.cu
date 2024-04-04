/* (c) mkbane@highendcompute.co.uk
   example of 1D grid to form A=B+C elemenwise
*/


#include <stdio.h>
#include <time.h>


__global__ void vecadd(float *B, float *C, float *A, int N)
{
  // form A = B + C
  int idx=blockIdx.x*blockDim.x+threadIdx.x;

  if (idx<N) {
    A[idx] = B[idx] + C[idx];
  }

}

#include <stddef.h>
#include <sys/time.h>
double get_wtime () {
/*
 * function to return double representing number of wall clock
 * seconds since some arbitrary point in time
 * mkbane (2023)
 * TO DO: update to clock_gettime 
 */
  struct timeval wallTime;
  gettimeofday(&wallTime, NULL);
  double seconds = wallTime.tv_sec;
  double microsecs = wallTime.tv_usec;
  return seconds + microsecs*1.0E-06;
}

void serial_vecadd(float *B, float *C, float *A, int N)
{
  // form A = B + C
  int idx;

  for (idx=0; idx<N; idx++) {
    A[idx] = B[idx] + C[idx];
  }

}

int main(int argc, char *argv[]) {
  int i, N;
  float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    if (argc<=1) {
      printf("%s requires 'N'\n", argv[0]);
      return -1;
    }
    N = atoi(argv[1]);

    // device timer
    cudaEvent_t startCuda, stopCuda;
    cudaEventCreate(&startCuda); cudaEventCreate(&stopCuda); // set them up, but not actually do timing

    int dSize = N*sizeof(float);
    cudaMalloc(&d_A, dSize);
    cudaMalloc(&d_B, dSize);
    cudaMalloc(&d_C, dSize);
    A = (float *) malloc(dSize);
    B = (float *) malloc(dSize);
    C = (float *) malloc(dSize);

    for (i=0; i<N; i++) {
      B[i] = 100.0*(rand() - rand())/RAND_MAX;  
      C[i] = 0.1 + rand()/RAND_MAX;
    }

    cudaMemcpy(d_B, B, dSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, dSize, cudaMemcpyHostToDevice);
    // see if any errors
    cudaError err = cudaGetLastError();
    if ( err != cudaSuccess) {
      printf("(1) CUDA RT error: %s \n",cudaGetErrorString(err));
    }

   // check (e.g.) 'deviceQuery' for device config maximums...

   int threadsPerBlock=32;
   int blocks = ceil((float)N/(float)threadsPerBlock);

   printf("For size %d, calling with %d blocks each of %d threads\n", N, blocks, threadsPerBlock);
   printf("1D grid => total of %d blocks, and total of %d threads\n",blocks,blocks*threadsPerBlock);

   cudaEventRecord(startCuda,0);
   vecadd <<<blocks, threadsPerBlock>>> (d_B, d_C, d_A, N);
   cudaEventRecord(stopCuda,0);

   // see if any errors launching/running kernel
   err = cudaGetLastError();
   if ( err != cudaSuccess) {
      printf("(2) CUDA RT error: %s \n",cudaGetErrorString(err));
   }


    cudaMemcpy(A, d_A, dSize, cudaMemcpyDeviceToHost);
    // see if any errors
    err = cudaGetLastError();
    if ( err != cudaSuccess) {
      printf("(3) CUDA RT error: %s \n",cudaGetErrorString(err));
    }


    // sample res so compiler not opt it all away
    i=N/2; 
    float eTime;
    cudaEventElapsedTime(&eTime, startCuda, stopCuda);
    printf("GPU: A[%d] = %f (cf %f) in %f milliseconds\n", i, A[i], B[i]+C[i], eTime);
    
    
    /* run on CPU */
    double start, finish;
    start = get_wtime();
    serial_vecadd (B, C, A, N);
    finish = get_wtime();
    printf("CPU: A[%d] = %f (cf %f) in %f milliseconds\n", i, A[i], B[i]+C[i], 1000.*(finish-start));

}
