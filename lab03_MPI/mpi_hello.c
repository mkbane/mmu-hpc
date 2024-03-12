/* 
   simple example for COMP328 & COMP528
   (c) Michael K Bane
   University of Liverpool, updated for MMU
*/

#include <stdio.h>
#include <mpi.h>

int main(void) {
  int myRank;   // unique 'rank' identify for each MPI process
  int numPEs;   // number of "processing elements" (i.e. MPI processes)

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];  
  int name_len;

  // first statement of an MPI code is to initialise the MPI
  MPI_Init(NULL,NULL);

  // obtain the 'processor' name
  MPI_Get_processor_name(processor_name, &name_len);

  // obtain the MPI process' unique "rank" from its MPI_COMM_WORLD "communicator"
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  // obtain number of MPI processes
  MPI_Comm_size(MPI_COMM_WORLD, &numPEs);

  printf("Hi from %s which has rank %d out of a total of %d processes\n", processor_name, myRank, numPEs);

  // let us set the variable X to be the rank of the processor, increment it and see what new value we have
  int X = myRank;
  X++;
  printf("Rank %d has X=%d\n",myRank, X);
  // i.e. "X" is local to each MPI process;
  // what we do to "X" on one process does not affect value on other MPI process
  
  // the last statement of an MPI is to finalise the MPI
  MPI_Finalize();
}
