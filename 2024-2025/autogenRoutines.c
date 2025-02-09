#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

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
int read_data(FILE *filePtr, int totalNum, double *x) {
  int numRead=0; // return the number of points successfully read from file
  // whereas totalNum is the value in file header as to number of data points
  for (int i=0; i<totalNum; i++) {
    // create random numbers between -100.0 and +100.0
    x[i] = -100.0 + 200.0*(double)rand() / (double)RAND_MAX;
    numRead++;
  }
  return numRead;
} // read_data()
