/* 
   (c) Michael K Bane
   MMU (Feb2024)
   version 1.0
*/

#include <stdio.h>
#include <omp.h>


int main(void) {
  int myThread;   // unique identify for each OpenMP thread
  int numPEs;     // number of "processing elements" (i.e. OpenMP threads)
  int X = -100;   
  int Y = +50;     
  
  // in a SERIAL region, print out X and thread info
  myThread = omp_get_thread_num();
  numPEs = omp_get_num_threads();
  printf("[%d of %d]: initial X=%d, Y=%d\n",myThread,numPEs,X,Y);

  X += 1;   // add one to X
  printf("[%d of %d]: and now X=%d, Y=%d\n",myThread,numPEs,X,Y);
  
  // start parallel region, add (10+thread number) to X & to Y and print everything
#pragma omp parallel default(none) private(myThread, numPEs, X) shared(Y)
  {
  myThread = omp_get_thread_num();
  X += 10 + myThread;
  Y += 10 + myThread;
  numPEs = omp_get_num_threads();
  printf("[%d of %d]: par reg X=%d, Y=%d\n",myThread,numPEs,X,Y);
  }

  printf("[%d of %d]: post par reg X=%d, Y=%d\n",myThread,numPEs,X,Y);

}
