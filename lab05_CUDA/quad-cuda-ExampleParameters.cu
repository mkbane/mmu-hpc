/*
 * A CUDA solution for quadrature
 * integral = sum of areas trapezoidals that approximate curve
 * area of trapezoidal = mean height * width
 * 
 * (c) michael k bane
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



// since we using static shared mem
#define maxThreads 1024

__global__
void calcMyArea(double x, double width, int numQuads, double *deviceBlockArea)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x; // thread num out of all threads across all blocks
  int p = threadIdx.x;                         // thread num on my given block
  //  int numBlocks = gridDim.x;
  int numThreads = blockDim.x;
  
  double myX;      // thread specific value of 'x'
  double fx, fxpw; // thread specific values of f(x) and f(x+width), respectively for given myX
  double height;        // mean height of rectangle
  
  if (numThreads > maxThreads) printf("exceeding static memory!\n");
  if (i < numQuads) {             // so if we have excess threads they do nothing
    __shared__ double myThreadArea[maxThreads]; // thread specific sub-area under curve
  
    // thread 0 per block to output info
    if (threadIdx.x == 0) {
      //DEBUG: printf("block %d has %d threads (cf %d quads)\n", blockIdx.x, numThreads, numQuads);
    }
    myThreadArea[p] = 0.0; // we accumulate area for given thread in this 'p' element of a shared area


    myX = x + i * width;
    // calc end points of rectangle and thus mean height
    // we hardwire the functions!
    fx = 20.4 + pow(myX,1.2)/3.1 - myX/log(3.0);
    fxpw = 20.4 + pow(myX+width,1.2)/3.1 - (myX+width)/log(3.0);
    height = 0.5*(fx+fxpw);
    myThreadArea[p] += height * width;
    //DEBUG: printf("block %d, thread %d with myX=%f now = %f\n", blockIdx.x, p, myX, myThreadArea[p]);

    // each thread now has a subarea, which we 'reduce' to give a block subarea saved in GPU shared memory
    // NB there are more efficient techniques available
    __syncthreads();
    if (p==0) {
      double myBlockArea = myThreadArea[0];
      // note that last block may not use all threads
      int myNumThreads = numThreads;
      if (blockIdx.x == gridDim.x - 1) {
	//DEBUG: printf("LAST BLOCK (#%d of %d) setting num threads to ", blockIdx.x, gridDim.x);
	myNumThreads = numQuads - blockIdx.x * blockDim.x;
	//DEBUG: printf("%d in block %d\n", myNumThreads, blockIdx.x);
      }
      //DEBUG: printf("block %d using %d threads in reduction step\n", blockIdx.x, myNumThreads);
      for (int t=1; t<myNumThreads; t++) {
	//DEBUG: printf("%f + %f = ", myBlockArea, myThreadArea[t]);
	myBlockArea += myThreadArea[t];
	//DEBUG: printf("%f\n", myBlockArea);
      }	
      // update GPU's global memory but only for this block's element
      //DEBUG: printf("blk updating element %d with value %f\n", blockIdx.x,myBlockArea);
      deviceBlockArea[blockIdx.x] = myBlockArea;
    }
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
    tpb = atoi(argv[4]);
  }
  else {
    printf("need to enter:\n %s a b numQuads threadsPerBlock\n", argv[0]);
    abort();
  }

  printf("Integrating from %f to %f using %d quads and %d threads per block.\nHold tight!\n", a,b,numberQuads,tpb);

  
  // sync device for reliable timing
  cudaDeviceSynchronize();
  double start = get_wtime();

  blks = (numberQuads + tpb - 1) / tpb;
  printf("using %d quads with %d blocks each using %d threads (total num threads: %d\n",
	 numberQuads, blks, tpb, blks*tpb);
  	
  const double width = (b-a) / (float) numberQuads;


  /* check have GPU else quit */
  cudaGetDeviceCount(&numGPUs);
  if (numGPUs >= 1 ) {
    printf("hello on CPU\n");
    /* call GPU kernel using b blocks and tpb threads per block  
     *
     * warning: the reduction only works for 1 block example
     * warning: do not amend value of 'b'
     *
     */

    // create variable array on device
    double *deviceBlockArea;
    cudaMalloc(&deviceBlockArea, blks * sizeof(double));

    cudaEvent_t startCuda, stopCuda;
    cudaEventCreate(&startCuda); cudaEventCreate(&stopCuda); // set them up, but not actually do timing
    cudaEventRecord(startCuda,0);

    calcMyArea<<<blks, tpb>>> (a, width, numberQuads, deviceBlockArea);

    cudaEventRecord(stopCuda,0);


    // check for device errors
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA RT error (post kernel) %s\n", cudaGetErrorString(err));

    // now get results
    double integrand;
    integrand = 0.0;
    //    double localSum[blks];
    double *localSum;
    localSum = (double *) malloc(blks * sizeof(double));

    cudaMemcpy(localSum, deviceBlockArea, blks * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i=0; i<blks; i++) {
      //DEBUG: printf("adding %f to integrand\n", localSum[i]);
      integrand += localSum[i];
    }	

    // check for device errors
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA RT error (post memcpy) %s\n", cudaGetErrorString(err));
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

