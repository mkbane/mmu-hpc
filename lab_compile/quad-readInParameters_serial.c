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
 * Version: serial, unoptimised
 * (c) michael k bane
 *
 *
 * v1.1 (08Feb2024): corrected loop for iteration space to go to '<numQ' 
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>  // for wallclock timing functions

double func(double x);   // prototype for func to integrate (separate file)

int main(int argc, char* argv[]) {
  double a, b;  /* bounds */
  int numberQuads;
  double integrand, width, x, y, meanHeight;
  int i;              /* loop counter */

  /* vars for timing */
  struct timeval wallStart, wallEnd;

  // read in from command line what to integrate
  // command line = exeName and 3 parameters (a, b, numberQuads) i.e. expect 4 items
  if (argc < 4) {
    printf("Need to pass bounds ('a','b') and numberQuads\n");
  }
  else {
    a = atof(argv[1]); // see 'man atof'
    b = atof(argv[2]); 
    numberQuads = atoi(argv[3]);
    printf("Integrating from a=%f to b=%f using %d trapezoidals\n",a,b,numberQuads);
  
    gettimeofday(&wallStart, NULL); // save start time in to variable 'wallStart'
    width = (b-a) / (float) numberQuads;
    integrand = 0.0;

    for (i=0; i<numberQuads; i++) {
      x = a + i*width;  // left of trapezoidal
      y = x + width;    // right of trapezoidal
      meanHeight = 0.5 * (func(x) + func(y));
      integrand += meanHeight*width;
    }

    gettimeofday(&wallEnd, NULL); // end time
    printf("(serial version): integral = %f\n", integrand);
    double wallSecs = (wallEnd.tv_sec - wallStart.tv_sec);           // just integral number of seconds
    double WALLtimeTaken = 1.0E-06 * ((wallSecs*1000000) + (wallEnd.tv_usec - wallStart.tv_usec)); // and now with any microseconds
    printf("WALL CLOCK Time: %f seconds  \n", WALLtimeTaken);
  }  
}

