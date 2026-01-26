/*
 * example calc of various statistics from input file
 * -- calcs are inline
 *
 * Requirements of file
 *  value of n
 *  followed by the n datapoints ('x')
 *  followed by n*n matrix ('A')
 *
 * (c) mkbane (Nov 2024 - Jan 2026)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>  // to ease timing

// function prototypes
// -- functions used in relative xxxRoutines.c
int get_num_data_points(FILE*);
int read_data(FILE*, int, double*, double*);  // returns number of points successfully read, populates arrays with data points

// main routine reading args from command line
int main(int argc, char** argv) {
  // data inputs
  int n;                 // number of data points
  double *x;             // pointer to 1D array (vector) data points
  double *A;             // pointer to 2D array (matrix) data points
  // variables required to determine stats of the data
  double *squaredDiffs;  // pointer to array holding squared differences (of x from mean of all x)
  // time total code (and elements thereof)
  double startTotalCode = omp_get_wtime();
  
  // access file here (and then pass pointer to file). This allows >1 routine to access same file.
  FILE* filePtr;
  char *filename = argv[1]; // filename is 1st parameter on command line
  filePtr = fopen(filename, "r"); // open file, given by sole parameter, as read-only
  if (filePtr == NULL) {
    printf("Cannot open file %s\n", filename);
  }
  else {
    int totalNum = get_num_data_points(filePtr);
    printf("There are allegedly %d data points to read\n", totalNum);
    x = (double *) malloc(totalNum * sizeof(double));
    A = (double *) malloc(totalNum*totalNum * sizeof(double));
    if (x == NULL | A == NULL) {
      // error in allocating memory
      printf("Error in allocating memory for data points ('x' and 'A')\n");
    }
    else {
      double start_readData = omp_get_wtime();
      n = read_data(filePtr, totalNum, x, A); // returns 'n' when x[0:n-1] and A[0:n-1][0:n-1] (or -1 if error)
      if (n == -1) {
	printf("error during read_data; aborting\n");
	return -1;
      }
      printf("%d data points successfully read [%f seconds]\n", n, omp_get_wtime()-start_readData);
      if (n != totalNum) printf("*** WARNING ***\n actual number read (%d) differs from header value (%d)\n\n",n,totalNum);
      squaredDiffs = (double *) malloc(n * sizeof(double));
      if (squaredDiffs == NULL) {
	// error in allocating memory
	printf("Error in allocating memory for squared differences\n");
      }
      else {
	/*
	 * main data processing loop
	 *
	 */
	printf("x[0]=%f\n", x[0]);
	double sum = 0.0;
	double mean;
	double start = omp_get_wtime();
	double local_start = omp_get_wtime();

	// (i) calc mean, variance
	// sum over x elements
	for (int i=0; i<n; i++) {
	  sum += x[i];
	}

	mean = sum/(double) n;

	// determine squares of differences
	double val;
	for (int i=0; i<n; i++) {
	  val = (x[i] - mean);
	  squaredDiffs[i] = val * val;
	}

	// sum over squares of differences
	sum = 0.0;
	for (int i=0; i<n; i++) {
	  sum += squaredDiffs[i];
	}	
	double variance = sum/(double) n;

	// (ii) calc min,max of absolute values
	// minimum absolute val
	double minabs = x[0];
#pragma omp parallel for default(none) shared(n, x) private(val) reduction(min:minabs)
	for (int i=0; i<n; i++) {
	  val = fabs(x[i]);
	  minabs = (val < minabs) ? val : minabs;
	}

	// maximum absolute val
	double maxabs = x[0];
#pragma omp parallel for default(none) shared(n, x) private(val) reduction(max:maxabs)
	for (int i=0; i<n; i++) {
	  val = fabs(x[i]);
	  maxabs = (val > maxabs) ? val : maxabs;
	}
	printf("Time for vector stats: %f seconds\n", omp_get_wtime()-local_start);
	// (iii) calc eigenvalue of matrix A
	printf("calculating eigenvalue");
	local_start = omp_get_wtime();
	// 'approx' iterates to eigenvectors 'z'
	double *approx;
	approx = (double *) malloc(n * sizeof(double));
	if (approx == NULL) {
	  // error in allocating memory
	  printf("Error in allocating memory for calculating eigenvalue\n");
	  return -100;
	}


	// set initial guess
	for (int i=0; i<n; i++) {
	  approx[i] = 1.0;
	}
	// iterate until error less than 'tol'
	double tol = 0.01;
	int iterCount=0;            // iteration count
	int i,j;                    // loop counters
	double z[n], zmax, err[n], errmax;
	printf(" to tolerance %f\n", tol);
	do {
	  iterCount++;
	  // DEBUG: printf("Iteration %d", iterCount);
	  double iterStart = omp_get_wtime();
	  // z = A*approx (matrix-vector multiplication)
	  for(i=0; i<n; i++) {
	    z[i]=0;
	    for(j=0; j<n; j++) {
	      int pos_ij = i + n*j;  // equivalent to A[i][j]
	      z[i]=z[i]+A[pos_ij]*approx[j];
	    }
	    //DEBUG printf("z[%d] = %g\n", i, z[i]);
	  }
	
	  // find max abs value of z
	  zmax=fabs(z[0]);
	  for(i=1; i<n; i++) {
	    if((fabs(z[i]))>zmax) {
	      zmax=fabs(z[i]);
	    }
	  }
	  //DEBUG printf("zmax = %g\n", zmax);
	  // normalise values of z
	  for(i=0; i<n; i++) {
	    z[i]=z[i]/zmax;
	  }

	  // calculate error between z and x
	  // (NB need allow for negative eigenvectors)
	  for(i=0; i<n; i++) {
	    err[i]=fabs((fabs(z[i]))-(fabs(approx[i])));
	    //DEBUG printf("err[%d] = %g\n", i, err[i]);
	  }

	  // find max error
	  errmax=err[0];
	  for(i=1; i<n; i++) {
	    if(err[i]>errmax)
	      errmax=err[i];
	  }
	  //DEBUG printf(" has error %g (%f seconds)\n", errmax, omp_get_wtime()-iterStart);
      
	  // set next iteration (guess) of x to be z
	  for(i=0; i<n; i++) {
	    approx[i]=z[i];
	    //DEBUG printf("new approx[%d] = %g\n", i, approx[i]);
	  }
	} while (errmax>tol);
	printf("Time for eigen stats: %f seconds\n", omp_get_wtime()-local_start);

	printf("Total stats wallclock time [%f seconds]\n", omp_get_wtime()-start);
	printf("Stats table\n");
	printf("Min abs val\tMax abs val\tMean\t\tStd Dev\n");
	printf("%f\t %f\t %f\t\t %f\n", minabs, maxabs, mean, sqrt(variance));
	printf("Eigen value of matrix: %f (%d iterations)\n", zmax, iterCount);
      } //memory alloc (squaredDiffs)
    } // memory alloc (x)
  } // file open
  printf("Completed (read+stats). [%f seconds]\n", omp_get_wtime()-startTotalCode);
} //main


