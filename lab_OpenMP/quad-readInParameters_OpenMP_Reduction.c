/*
 * Code to numerically integrate ("quadrature") function in file compiled with this that contains 
 *   function func(double x)
 * from a to b using numberQuads, where
 * a,b, numberQuads 
 * given on command line
 * 
 * integral = sum of areas trapezoidals that approximate curve
 * area of trapezoidal = mean height * width
 *
 * Version: OpenMP, Reduction
 * (c) michael k bane
 *
 *
 * v1.0 (13Feb2024)
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>  // OpenMP prototypes

double func(double x);      // prototype for func to integrate (separate file)

int main(int argc, char* argv[]) {
  double a, b;  /* bounds */
  int numberQuads;
  double integrand, width, x, y, meanHeight;
  int i;              /* loop counter */

  /* OpenMP vars for timing */
  double wallStart, wallEnd;

  // OpenMP variables
  int myThread, numPEs;

  // read in from command line what to integrate
  // command line = exeName and 3 parameters (a, b, numberQuads) i.e. expect 4 items
  if (argc < 4) {
    printf("Need to pass bounds ('a','b') and numberQuads\n");
  }
  else {
    a = atof(argv[1]); // see 'man atof'
    b = atof(argv[2]); 
    numberQuads = atoi(argv[3]);
    printf("Integrating from a=%f to b=%f using %d trapezoidals\n",a,b,numberQuads);
    wallStart = omp_get_wtime();
    width = (b-a) / (float) numberQuads;
    integrand = 0.0;

    // OpenMP implementation based on this being a Reduction problen
    // shared: variables only read from
    // private: variables with temp values (within par reg) with >1 thread updating
    // then consider var "numPEs". Put as shared so have access after parallel region
    // we are reducing partial values of integrand (on each thread) to global value of integrand (on controller thread)
#pragma omp parallel default(none) \
  shared(numberQuads, a, width)	 \
  private(myThread, i, x, y, meanHeight) \
  shared(numPEs) \
  reduction(+: integrand)
    {
    myThread = omp_get_thread_num();
    numPEs = omp_get_num_threads();  // no race condition since same value would be written by each thread
    #pragma omp for
    for (i=0; i<numberQuads; i++) {
      x = a + i*width;  // left of trapezoidal
      y = x + width;    // right of trapezoidal
      meanHeight = 0.5 * (func(x) + func(y));
      integrand += meanHeight*width;
    } // end for
    } // end of parallel region

    wallEnd = omp_get_wtime();
    printf("(Openmp [Reduce] version using %d threads): integral = %f\n", numPEs, integrand);
    double WALLtimeTaken = wallEnd - wallStart;
    printf("WALL CLOCK Time: %f seconds  \n", WALLtimeTaken);
  }  
}

