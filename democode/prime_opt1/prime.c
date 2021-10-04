#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "prime.h"

void computePrimes(char results[], bignum s, bignum n){
   
   bignum i;

   for(i=s; i< s+n; i++){
      results[i]=isPrime(i);
   }
}


/** This is the thread function that is passed to pthread_create */
void * primeThread(void *args){

   jobSpec *js = (jobSpec *) args;
   char * results = js->results;
   bignum start = js->start;
   bignum max = js->max;
   int tid = js->tid;
   bignum j = (bignum)tid;

   // we could free either js or args here, since
   // they are really the same pointer. In any case, we
   // are freeing the memory occupied by the jobSpec.
   free(js);

   //printf("Computing %llu primes starting at %llu\n", n, start);
   for(; j <= max; j =  j + NTHREADS )
   {
       results[j] = isPrime(j); 
       //isPrime(j); //we find the compuation on the CPU takes a very small percent of total parallel execution time cost  
   } 
   
   return NULL;

}


/** Sets each element results[i] to 0 or 1. If s+i is prime, 
  * results[i] isset to 1, and set to 0 otherwise. There should be
  * n elements in the results array.
  */
int pcomputePrimes(char results[], bignum s, bignum n){

   pthread_t threads[NTHREADS];
   int i;
   jobSpec *js;
   
   
   // Create a job spec for each thread, and launch the thread.
   for(i=0; i< NTHREADS; i++){
   
      js = malloc(sizeof(jobSpec));

      js->results=results;
      js->start = s;
      js->max = n;
      js->tid = i;
   
      if (pthread_create(&(threads[i]), NULL, primeThread, js) != 0){
      
         return EXIT_FAILURE;
      }
   }
   
   // Wait for the threads to finish.
   for(i=0; i< NTHREADS; i++){
   
      if (pthread_join(threads[i], NULL) != 0){
         return EXIT_FAILURE;
      }
      
   }
   
   return EXIT_SUCCESS;
}

int isPrime(bignum x){

   bignum i;
   bignum lim = (bignum) sqrt(x) + 1;
      
   for(i=2; i<lim; i++){
      if ( x % i == 0)
         return 0;
   }

   return 1;
}


void initializeArray(char a[], bignum len){

   int i;
   
   for(i=0; i<len; i++){
      a[i]= i;
   }

}

void printArray(char a[], int len){

   int i;
   
   for(i=0; i<len; i++){
   
      printf("%d", a[i]);

   }

}

int arrSum( char a[], bignum len )
{
    int i, s = 0;
    for( i = 0; i < len; i ++ )
        s += a[i];

    return s;
}
