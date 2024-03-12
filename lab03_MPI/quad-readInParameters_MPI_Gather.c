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
 * Version: MPI (Gather) - better alternatives exist
 * (c) michael k bane
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>       // MPI prototypes & constants

double func(double x);   // prototype for func to integrate (separate file)

int main(int argc, char* argv[]) {
  double a, b;  /* bounds */
  int numberQuads;
  double integrand, width, x, y, meanHeight;
  int i;              /* loop counter */

  /* MPI vars for timing */
  double wallStart, wallEnd;

  // MPI variables //
  int myRank, numPEs;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numPEs);
  // each MPI process reads in from command line what to integrate
  // command line = exeName and 3 parameters (a, b, numberQuads) i.e. expect 4 items
  if (argc < 4) {
    if (myRank == 0) printf("Need to pass bounds ('a','b') and numberQuads\n");
  }
  else {
    a = atof(argv[1]); // see 'man atof'
    b = atof(argv[2]); 
    numberQuads = atoi(argv[3]);
    if (myRank == 0) {
      printf("Integrating from a=%f to b=%f using %d trapezoidals\n",a,b,numberQuads);
      wallStart = MPI_Wtime();
    }
    width = (b-a) / (float) numberQuads;
    integrand = 0.0;

    int myStart, myFinish, myNum;
    myNum = numberQuads/numPEs;
    myStart = myRank * myNum;
    myFinish = myStart + myNum;
    if (myRank == numPEs-1) myFinish = numberQuads; // this could be improved
    //DEBUG: printf("[%d] start=%d finish=%d\n", myRank, myStart, myFinish);
    double myArea = 0.0;
    
    for (i=myStart; i<myFinish; i++) {
      //DEBUG: printf("[%d] doing i=%d\n",myRank, i);
      x = a + i*width;  // left of trapezoidal
      y = x + width;    // right of trapezoidal
      meanHeight = 0.5 * (func(x) + func(y));
      myArea += meanHeight*width;
    }
    //DEBUG: printf("[%d] myArea=%g\n", myRank, myArea);
    // at this point we have "myArea" for each MPI process
    // we need to sum this together
    // one way: bring all "myArea" values onto rank=0 process and sum there
    // Using MPI_Gather we bring a single variable from each process together into an array"ValsToSum" on rank=0
    double ValsToSum[numPEs];
    MPI_Gather(&myArea, 1, MPI_DOUBLE,      // equivalent to each process sending its "myArea"
	       ValsToSum, 1, MPI_DOUBLE,    // we receive 1 variable (a double) from each process into ValsToSum
	       0, MPI_COMM_WORLD);          // receive to ValsToSum on rank=0 only

    // sum
    for (int i=0; i<numPEs; i++) {
      integrand += ValsToSum[i];
    }

    if (myRank == 0) {
      wallEnd = MPI_Wtime();
      printf("(MPI version using %d processes): integral = %f\n", numPEs, integrand);
      double WALLtimeTaken = wallEnd - wallStart;
      printf("WALL CLOCK Time: %f seconds  \n", WALLtimeTaken);
    }
  }
  MPI_Finalize();
}

