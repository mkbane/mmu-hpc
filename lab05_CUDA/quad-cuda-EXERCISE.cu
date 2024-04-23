/*
 * A CUDA solution for quadrature (each thread determines area of one trapezoidal; summation on CPU)
 * integral = sum of areas trapezoidals that approximate curve
 * area of trapezoidal = mean height * width
 * 
 * THis is a PARTIAL SOLUTION - see lab worksheet & complete all "TO DO" sections below
 * 
 * (c) michael k bane (2024)
 */

#include <math.h>
#include <stdlib.h>
#include <sys/time.h>  // for wallclock timing functions
#include <stdio.h>
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


// this CUDA kernel calculates area of a quadrangle on each thread
__global__
void calcMyArea(double a, double width, int numQuads, double *quadArea)
{

  // TO DO: add expression go calculate "i" as unique index based upon #threadsPerBlock, #blocks and threadIdx
  int i = 
  
  double myX;      // thread specific value of 'x'
  double fx, fxpw; // thread specific values of f(x) and f(x+width), respectively for given myX
  double height;        // mean height of rectangle

  // TO DO: add boolean expression to ensure excess  threads  do nothing
  if (...) {             
  
    // thread 0 per block to output info
    if (threadIdx.x == 0) {
      //DEBUG: printf("block %d has %d threads (cf %d quads)\n", blockIdx.x, numThreads, numQuads);
    }

    myX = a + i * width;
    // calc end points of rectangle and thus mean height
    // we hardwire the functions!
    fx = 20.4 + pow(myX,1.2)/3.1 - myX/log(3.0);
    fxpw = 20.4 + pow(myX+width,1.2)/3.1 - (myX+width)/log(3.0);
    height = 0.5*(fx+fxpw);
    quadArea[i] = // TO DO: add expression for area of rectangle based on height and width
  }
  else  {
    //DEBUG: printf("thread %d does nothing \n", i);
  }
}



int main(int argc, char *argv[]) {
  int numGPUs;
  double a,b; // bounds (user input)
  
  
  int numberQuads, blks, tpb;
  
  // parse input args
  if (argc==5) {
    a = atof(argv[1]);
    b = atof(argv[2]);
    numberQuads = atoi(argv[3]);
    tpb = atoi(argv[4]);  // user defines the number of threads-per-block
  }
  else {
    printf("need to enter:\n %s a b numQuads threadsPerBlock\n", argv[0]);
    abort();
  }

  printf("Integrating from %f to %f using %d quads and %d threads per block.\nHold tight!\n", a,b,numberQuads,tpb);

  // sync device for reliable timing
  cudaDeviceSynchronize();
  double start = get_wtime();

  // calculate number of blocks required
  blks = (numberQuads + tpb - 1) / tpb;
  printf("using %d quads with %d blocks each using %d threads (total num threads: %d\n",
	 numberQuads, blks, tpb, blks*tpb);
  	
  const double width = (b-a) / (float) numberQuads;


  /* check have GPU else quit */
  cudaGetDeviceCount(&numGPUs);
  if (numGPUs >= 1 ) {
    printf("hello on CPU\n");

    // create variable arrays on host & on device
    long quadArea_numBytes = numberQuads * sizeof(double);
    printf("allocating quadArea (%ld bytes)\n", quadArea_numBytes);
    double *quadArea;    // host
    double *d_quadArea;  // device
    quadArea = (double *) malloc(quadArea_numBytes);  // alloc on CPU memory
    if (quadArea == NULL) {
      printf("error alloc quadArea (%ld bytes)\n", quadArea_numBytes);
      abort();
    }
    cudaMalloc(&d_quadArea, quadArea_numBytes);  // alloc on GPU global memory

    cudaEvent_t startCuda, stopCuda;
    cudaEventCreate(&startCuda); cudaEventCreate(&stopCuda); // set them up, but not actually do timing
    cudaEventRecord(startCuda,0);


    // launch kernel
    // TO DO: complete the next line to launch relevant number of blocks and threads-per-block
    calcMyArea    (a, width, numberQuads, d_quadArea);

    cudaEventRecord(stopCuda,0);


    // check for device errors
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA RT error (post kernel) %s\n", cudaGetErrorString(err));

    // pull all the data for each area of each quadrangle from the GPU
    cudaMemcpy(quadArea, d_quadArea, quadArea_numBytes, cudaMemcpyDeviceToHost);
    // sum all the areas to find integrand
    double integrand = 0.0;
    for (int i=0; i<numberQuads; i++) {
      //DEBUG: printf("i=%d: adding %f to integrand\n", i, quadArea[i]);
      integrand += quadArea[i];
    }	

    cudaDeviceSynchronize();
    double finish = get_wtime();
    printf("GPU [naive CUDA] integral: %f\n", integrand);
    // determine elapsed time of kernel only
    float eTime;
    cudaEventElapsedTime(&eTime, startCuda, stopCuda);
    printf("%d tpb: WALL CLOCK Time: %f seconds (kernel: %f seconds)\n",tpb,(finish-start), eTime/1000.0);


  }

  else {
    printf("no GPU present\n");
  }

}

