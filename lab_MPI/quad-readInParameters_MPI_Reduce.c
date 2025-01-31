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
 * Version: MPI (Reduce)
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
  // eacho MPI process reads in from command line what to integrate
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
    // this is a mathematical/computational pattern known as "reduction"
    // we use the MPI_Reduce to (do functional equivalent of) bring each "myArea" from each process together, summing them together into integrand on rank=0

    MPI_Reduce(&myArea,      // equivalent to each process sending its "myArea"
	       &integrand,   // to be summed (see 'operation') into "integrand"
	       1, MPI_DOUBLE,    // myArea and integrand are scalars, i.e. of length 1
	       MPI_SUM, 0,          // the 'operation' is to sum values from all ranks into "integrand" on rank=0 only
	       MPI_COMM_WORLD);
    
    if (myRank == 0) {
      wallEnd = MPI_Wtime();
      printf("(MPI version using %d processes): integral = %f\n", numPEs, integrand);
      double WALLtimeTaken = wallEnd - wallStart;
      printf("WALL CLOCK Time: %f seconds  \n", WALLtimeTaken);
    }
  }
  MPI_Finalize();
}

