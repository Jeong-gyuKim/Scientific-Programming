#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include "tests.h"
#include <cmath>

#ifdef SPRNG_MPI
#include <mpi.h>
#endif

using namespace std;

void init_collision (long n, int logmd, int logd);
long collision (long n, int logmd, int logd);
double identity (double d);
void compute_probability (double *A, int n, int m);

#define MIN_CHISQ_NTESTS 50

int *theArray, *mask;
int arraysize, intsize;
double *probability;

int main (int argc, char *argv[])
{
  long ntests, n, i, *bins, ncollisions;
  double *V, result;
  int logmd, logd, Bins_used;

  bins = NULL;
  V = NULL;

  if (argc != N_STREAM_PARAM + 4)
    {
      fprintf (stderr, "USAGE: %s (... %d arguments)\n", argv[0],
      N_STREAM_PARAM + 3);
      exit (1);
    }

  ntests = init_tests (argc, argv);

  n = atol (argv[N_STREAM_PARAM + 1]);
  logmd = atoi (argv[N_STREAM_PARAM + 2]);
  logd = atoi (argv[N_STREAM_PARAM + 3]);

  if (logmd * logd > 31)
    {
      fprintf (stderr, "ERROR: log(m-d)*log(d) = %d must be less than 32\n",
	       logmd * logd);
      exit (1);
    }
  if ((1 << (logmd * logd)) < n)
    {
      fprintf (stderr, "ERROR: m = %d must be at least as high as n = %ld\n",
	       (1 << (logmd * logd)), n);
      exit (1);
    }

  if (NTESTS < MIN_CHISQ_NTESTS)
    V = new double[NTESTS];
  else
    {
      bins = new long[n + 1];
      memset (bins, 0, (n + 1) * sizeof(long));
    }

  init_collision (n, logmd, logd);

  for (i = 0; i < ntests; i++)
    {
      ncollisions = collision (n, logmd, logd);
      if (NTESTS < MIN_CHISQ_NTESTS)
	V[i] = probability[ncollisions];
      else
	{
	  bins[ncollisions]++;
	}

      next_stream ();
    }

#if defined(SPRNG_MPI)
  if( NTESTS < MIN_CHISQ_NTESTS) 
    getKSdata(V,NTESTS);
  else
    reduce_sum_long(bins,n+1);
#endif

  if (proc_rank == 0 && NTESTS < MIN_CHISQ_NTESTS)
    {
      result = KS (V, NTESTS, identity);
      printf ("\nResult: KS value = %f", result);

      /*for(i=0; i<NTESTS; i++)
       printf("\tstream = %d, prob = %f\n", i, V[i]);*/

      result = KSpercent (result, NTESTS);
      printf ("\t %% = %.2f\n\n", result * 100.0);
    }
  else if (proc_rank == 0)
    {
      printf (
	  "\n Please ignore any warning message about the effect of combining bins\n\n");

      result = chisquare (bins, probability, NTESTS, n + 1, &Bins_used);
      printf ("\nResult: Chi Square value = %f", result);

      /*for(i=0; i<n; i++)
       printf("\t# of collisions = %d, frequency = %ld\n", i, bins[i]);*/

      result = chipercent (result, Bins_used - 1);
      printf ("\t %% = %.2f\n\n", result * 100.0);
    }

  delete[] mask;
  delete[] probability;
  delete[] theArray;

  if (NTESTS < MIN_CHISQ_NTESTS)
    delete[] V;
  else
    delete[] bins;

#if defined(SPRNG_MPI)
  MPI_Finalize();
#endif
  return 0;
}

double identity (double d)
{
  return d;
}

void init_collision (long n, int logmd, int logd)
{
  long m, size;
  int tempmask, i;

  intsize = sizeof(int) * 8; /*** Assumes 8 bit characters ****/
  m = (1 << (logmd * logd));
  size = m / intsize;
  if (m % intsize > 0)
    size++;

  arraysize = size;
  theArray = new int[arraysize];
  mask = new int[intsize];

  tempmask = 1;

  for (i = 0; i < intsize; i++)
    {
      mask[i] = tempmask;
      tempmask <<= 1;
    }

  probability = new double[n + 1];

  compute_probability (probability, (int) n, m);
}

void compute_probability (double *A, int n, int m)
{
  int i, j, j0, j1;

  for (j = 0; j <= n; j++)
    A[j] = 0.0;

  A[1] = j0 = j1 = 1;

  for (i = 0; i < n - 1; i++)
    {
      j1++;
      for (j = j1; j >= j0; j--)
	{
	  A[j] = (double) j / (double) m * A[j]
	      + ((1.0 + 1.0 / (double) m) - (double) j / (double) m) * A[j - 1];

	  if (A[j] < 1.0e-20)
	    {
	      A[j] = 0.0;
	      if (j == j1)
		j1--;
	      else if (j == j0)
		j0++;
	    }
	}
    }

  if (NTESTS < MIN_CHISQ_NTESTS
      && sqrt ((double) NTESTS / (double) (j1 - j0 + 1)) > 0.5)
    {
      fprintf (stderr, "WARNING: Error in KS estimate may be ~ %f\n",
	       sqrt ((double) NTESTS / (double) (j1 - j0 + 1)));
      if (NTESTS * n > 1.0e10) /* We cannot afford to waste large amounts ... */
	{ /* ... of computer time */
	  fprintf (stderr, "Exiting ... \n");
	  exit (-1);
	}
    }

  if (NTESTS < MIN_CHISQ_NTESTS)
    {
      for (j = j1 - 1; j >= j0; j--)
	A[j] += A[j + 1];

      if (A[j0] >= 1.0)
	A[j0] = 1.0 - 1.0e-7; /* returning 1.0 confuses the KS test! */
    }

}

long collision (long n, int logmd, int logd)
{
  long m, i, ncollisions;
  int d, j, index, bit;
  unsigned int temp, num;

  d = 1 << logd;
  m = 1 << (logmd * logd);

  ncollisions = 0;
  memset (theArray, 0, arraysize * sizeof(int));

  for (i = 0; i < n; i++)
    {
      temp = 0;
      for (j = 0; j < logmd; j++)
	{
	  num = static_cast<unsigned int> (d * get_rn ());
	  temp <<= logd;
	  temp |= num;
	}

      index = temp / intsize;
      bit = temp % intsize;

      if (theArray[index] & mask[bit])
	ncollisions++;
      else
	theArray[index] |= mask[bit];
    }

  /*printf("ncollisions = %ld, probability = %f\n", ncollisions, probability[n-ncollisions]);*/

  return n - ncollisions;
}
