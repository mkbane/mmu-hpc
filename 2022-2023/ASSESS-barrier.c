/*
 * MPI solution for quadrature (block, pt2pt, non-verbose)
 * integral = sum of areas trapezoidals that approximate curve
 * area of trapezoidal = mean height * width
 *
 * (c) michael k bane
 */

#include <stdio.h>
#include <math.h>

#include <sys/time.h>  // for wallclock timing functions
#include <mpi.h>       // header for MPI 

double func(double x);   // prototype for func to integrate (separate file)

int main(void) {
  const double a=0.1, b=500.1;  /* bounds */
  const int numberQuads = 4096000;
  double integrand, width, x, y, meanHeight;
  int i;              /* loop counter */

  /* for timing */
  struct timeval wallStart, wallEnd;

  /* vars for MPI */
  int numProcesses, rankNum;


  MPI_Init(NULL, NULL);

  // barrier to ensure we do not time MPI_Init
  MPI_Barrier(MPI_COMM_WORLD);
  
  
  gettimeofday(&wallStart, NULL); // save start time in to variable 'wallStart'
  width = (b-a) / (float) numberQuads;
  integrand = 0.0;

  /* MPI explicitly split iterations over MPI processes */
  MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankNum);

  int numberQuads_perProcess = numberQuads/numProcesses;
  int start_perProcess = rankNum*numberQuads_perProcess;
  if (rankNum == numProcesses-1) numberQuads_perProcess = numberQuads - (numberQuads_perProcess*rankNum);
  int finish_perProcess = start_perProcess + numberQuads_perProcess - 1;
   
  /* by having different sub-ranges of the iteration space, 
     we have divided the work between the MPI processes */
  double localSum = 0.0;
  for (i=start_perProcess; i<=finish_perProcess; i++) {
    x = a + i*width;  // left of trapezoidal
    y = x + width;    // right of trapezoidal
    meanHeight = 0.5 * (func(x) + func(y));
    localSum += meanHeight*width;
  }
  // debugging: next line will show the accumulate localSum per MPI process
  // printf("MPI partial sum on %d rank of %d processes - integral = %f\n", rankNum, numProcesses, localSum);

  /* now we have to sum all 'localSum' from each rank in to a global "sum"
     this is known as a REDUCTION */
  MPI_Reduce(&localSum, &integrand, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rankNum == 0) {
    gettimeofday(&wallEnd, NULL); // end time
    printf("MPI- integral = %f\n", integrand);
    double wallSecs = (wallEnd.tv_sec - wallStart.tv_sec);           // just integral number of seconds
    double WALLtimeTaken = 1.0E-06 * ((wallSecs*1000000) + (wallEnd.tv_usec - wallStart.tv_usec)); // and now with any microseconds
    printf("WALL CLOCK Time: %f seconds  \n", WALLtimeTaken);
  }


  MPI_Finalize();
}

// function to integrate
double func (double x) {
  return pow(x,1.5)/3.1 - x/log(3.0);
}

