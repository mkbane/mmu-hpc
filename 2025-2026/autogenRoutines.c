/*
 * code to simulate reading in data (airport data off kaggle)
 * n, vector, matrix
 * (i) read in n from file
 * (ii) auto-generate vector[0:n-1] and matrix[0:n-1][0:n-1]
 * matrix is stored as 1D array
 *
 * (c) mkbane (Jan 2026)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

// define RNG data
#define MIN_VALUE -10.0

/*
 * getting data specific to aero data
 */
int get_num_data_points(FILE *filePtr) {
#include <string.h>
  // open file, get initial row
  int totalNum = 0;
  char buffer[BUFSIZ];  // max number to read
  fgets(buffer, BUFSIZ-1, filePtr);
  sscanf(buffer, "%d", &totalNum);             // scan buffer and convert to an int, saving into var 'n'
  printf("we have %d data points to read\n", totalNum);

  return totalNum;
}


/*
 * read_data: version to autogen data
 *
 */
int read_data(FILE *filePtr, int totalNum, double *vector, double *matrix) {
  double range = 2.0 * fabs(MIN_VALUE);
  double normalise = range/(double) RAND_MAX;
  printf("range %f: from %f to %f\nnormalise: %g\nRAND_MAX: %d\n", range, MIN_VALUE, fabs(MIN_VALUE), normalise, RAND_MAX);
  int pos_ij;
  for (int i=0; i<totalNum; i++) {
    // create random numbers between -100.0 and +100.0
    vector[i] = rand()*normalise + MIN_VALUE;
    double sum = 0.0;
    for (int j=0; j<totalNum; j++) {
      pos_ij = i + j*totalNum;
      matrix[pos_ij] = rand()*normalise + MIN_VALUE;
      if (j != i) sum += fabs(matrix[pos_ij]);
    }
    // set diagonal element to greater than sum of absolute values of other elements
    pos_ij = i + i*totalNum;
    matrix[pos_ij] = sum * 1.0001;
  }
  return totalNum;
} // read_data()
