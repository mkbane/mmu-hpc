/*
 * example calc of variance
 * -- var calcs inline
 *
 * Requirements
 * 1. file contains n, followed by the n datapoints
 *
 * mkbane (Nov 2024)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>  // to ease timing

// function prototypes
// -- functions used in relative xxxRoutines.c
int get_num_data_points(FILE*);
int read_data(FILE*, int, double*);  // returns number of points successfully read, populates array with data points

// main routine reading args from command line
int main(int argc, char** argv) {
  int n;                 // number of data points
  double *x;             // pointer to array holding data points
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
    if (x == NULL) {
      // error in allocating memory
      printf("Error in allocating memory for data points\n");
    }
    else {
      double start_readData = omp_get_wtime();
      n = read_data(filePtr, totalNum, x); // this is actual number of points read
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

	// sum over x elements
	for (int i=0; i<n; i++) {
	  sum += x[i];
	}
	//DEBUG: printf("sum of x values: %f [%f seconds]\n", sum, omp_get_wtime()-local_start);

	mean = sum/(float) n;
	//DEBUG: printf(" with mean: %f\n", mean);

	// determine squares of differences
	//DEBUG: local_start = omp_get_wtime();
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
	//DEBUG: printf("sum of squared differences: %f [%f seconds]\n", sum, omp_get_wtime()-local_start);

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


	printf("Total wallclock time [%f seconds]\n", omp_get_wtime()-start);
	printf("min, max absolute values are: %f, %f\n", minabs, maxabs);
	double variance = sum/(float) n;
	printf(" with mean: %f\n", variance);
	printf("The variance is %f\n", variance);
      } //memory alloc (squaredDiffs)
    } // memory alloc (x)
  } // file open
  printf("Completed. [%f seconds]\n", omp_get_wtime()-startTotalCode);
} //main


