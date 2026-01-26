#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// function prototypes
int get_num_data_points(FILE*);
int read_data(FILE*, int, double*);

int getData(char *filename, double *x)
{
  int rc = -1;
  FILE* filePtr;
  filePtr = fopen(filename, "r"); // open file, given by sole parameter, as read-only
  if (filePtr == NULL) {
    printf("Cannot open file %s\n", filename);
    rc = 0;
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
      int n = read_data(filePtr, totalNum, x); // this is actual number of points read
      printf("%d data points successfully read [%f seconds]\n", n, omp_get_wtime()-start_readData);
      if (n != totalNum) printf("*** WARNING ***\n actual number read (%d) differs from header value (%d)\n\n",n,totalNum);
      rc = n;
    }
  }
  return rc;
}

/*
 * getting data specific to aero data
 */
int get_num_data_points(FILE *filePtr) {
#include <string.h>
  // open file, get initial row
  int totalNum = 0;
  char buffer[BUFSIZ];  // for this example, max chars per line
  fgets(buffer, BUFSIZ-1, filePtr);
  sscanf(buffer, "%d", &totalNum);             // scan buffer and convert to an int, saving into var 'n'
  printf("we have %d data points to read\n", totalNum);

  return totalNum;
}


/*
 * read_data: version to read 9th real number each line of csv
 *
 */
int read_data(FILE *filePtr, int totalNum, double *x) {
  int numRead=0; // number of points successfully read from file
  // whereas totalNum is the value in file header as to number of data points
  
  // read each rows remaining in file up to max of totalNum (else we will get memory error)
  char buffer[BUFSIZ];  // for this example, max chars per line

  /*
   * for each row, read 9th value per line of csv file
   *
   */
  char *dummyString,*dummyString1,*priceString; // max length of any string
  char remainder[256]; // max length of remainder of line of strings
  // loop whilst more rows to process
  while (fgets(buffer, BUFSIZ-1, filePtr) != NULL && numRead<totalNum) {
    //DEBUG printf("%d %d\n", numRead, totalNum);
    //DEBUG printf("buffer: %s\n", buffer);
    dummyString1=strtok(buffer, ",");
    //DEBUG printf("discarind string: %s\n", dummyString1);
    // loop over next 7 strings
    for (int i=0; i<7; i++) {
      dummyString=strtok(NULL, ",");
      //DEBUG printf("discarind string: %s\n", dummyString);
    }
    // 9th token is our totalPrice which we use as our data points
    priceString = strtok(NULL, ",");
    x[numRead] = atof(priceString);
    //DEBUG printf("x[%d]=%f\n",numRead,x[numRead]);
    numRead++;
  } // while
  printf("Read %d pairs of points.\n", numRead);
  return numRead;
} // read_data()
