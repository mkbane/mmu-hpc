/* 
   (c) Michael K Bane
   MMU (Feb2024)
   version 1.0
*/

#include <stdio.h>
#include <omp.h>

int main(void) {
  int myThread;   // unique identify for each OpenMP thread
  int numPEs;   // number of "processing elements" (i.e. OpenMP threads)
  int X = -100;
  
  // in a SERIAL region, print out X and thread info
  myThread = omp_get_thread_num();
  numPEs = omp_get_num_threads();
  printf("SERIAL: on thread #%d of %d, X=%d\n",myThread,numPEs,X);

  // start parallel region (with no explicit data sharing), print out X and thread info
#pragma omp parallel
  {
  myThread = omp_get_thread_num();
  numPEs = omp_get_num_threads();
  printf("PAR (default data clauses): on thread #%d of %d, X=%d\n",myThread,numPEs,X);
  }
  
  // in a SERIAL region, print out X and thread info
  myThread = omp_get_thread_num();
  numPEs = omp_get_num_threads();
  printf("SERIAL: on thread #%d of %d, X=%d\n",myThread,numPEs,X);

  // start parallel region (with correct data sharing), print out X and thread info
#pragma omp parallel default(none) private(myThread, numPEs, X)
  {
  myThread = omp_get_thread_num();
  numPEs = omp_get_num_threads();
  printf("PAR (all private): on thread #%d of %d, X=%d\n",myThread,numPEs,X);
  }

  // in a SERIAL region, print out X and thread info
  myThread = omp_get_thread_num();
  numPEs = omp_get_num_threads();
  printf("SERIAL: on thread #%d of %d, X=%d\n",myThread,numPEs,X);

  // start parallel region (with correct data sharing), print out X and thread info
#pragma omp parallel default(none) private(myThread, numPEs) shared(X)
  {
  myThread = omp_get_thread_num();
  numPEs = omp_get_num_threads();
  printf("PAR (X shared): on thread #%d of %d, X=%d\n",myThread,numPEs,X);
  }

}
