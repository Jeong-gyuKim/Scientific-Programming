#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "primes_32.h"
#include "primelist_32.h"
using namespace std;

#define YES 1
#define NO  0
#define NPRIMES 1000
#define primes primes32
int primes[NPRIMES];

int init_prime_32 (void)
{
  int i, j, obtained = 0, isprime;
  for (i = 3; i < MINPRIME; i += 2)
  {
    isprime = YES;
    for (j = 0; j < obtained; j++)
   if (i % primes[j] == 0)
   {
    isprime = NO;
    break;
   }
  else if (primes[j] * primes[j] > i)
    break;
  if (isprime == YES)
  {
    primes[obtained] = i;
    obtained++;
  }
  }
  return obtained;
}

int getprime_32 (int need, int *prime_array, int offset)
{
  static int initiallized = NO, num_prime;
  int largest;
  int i, isprime, index, obtained = 0;
  if (need <= 0)
  {
    fprintf (stderr,
            "WARNING: Number of primes needed = %d < 1; None returned\n",
            need);
    return 0;
  }

  if (offset < 0)
  {
    fprintf (stderr,
            "WARNING: Offset of prime = %d < 1; None returned\n",
            offset);
    return 0;
  }
  if (offset + need - 1 < PRIMELISTSIZE1)
  {
    memcpy (prime_array, prime_list_32 + offset, need * sizeof(int));
    return need;
  }
  if (!initiallized)
  {
    num_prime = init_prime_32 ();
    largest = MAXPRIME;
    initiallized = YES;
  }
  if (offset > MAXPRIMEOFFSET)
  {
    fprintf (stderr,
            "WARNING: generator has branched maximum number of times;\nindependence of generators no longer guaranteed");
            offset = offset % MAXPRIMEOFFSET;
  }
  if (offset < PRIMELISTSIZE1) /* search table for previous prime */
  {
    largest = prime_list_32[offset] + 2;
    offset = 0;
  }
  else
  {
    index = (int) ((offset - PRIMELISTSIZE1 + 1) / STEP) + PRIMELISTSIZE1 - 1;
    largest = prime_list_32[index] + 2;
    offset -= (index - PRIMELISTSIZE1 + 1) * STEP + PRIMELISTSIZE1 - 1;
  }
  while (need > obtained && largest > MINPRIME)
  {
   isprime = YES;
   largest -= 2;
   for (i = 0; i < num_prime; i++)
   if (largest % primes[i] == 0)
  {
    isprime = NO;
    break;
  }
    if (isprime == YES && offset > 0)
 offset--;
    else if (isprime == YES)
 prime_array[obtained++] = largest;
  }
  if (need > obtained)
    fprintf (stderr,
             "ERROR: Insufficient number of primes: needed %d, obtained %d\n",
	         need, obtained);
  return obtained;
}

#if 0
main()
{
  int newprimes[1500], np, i;
  np = getprime_32(2,newprimes,0);
  np = getprime_32(2,newprimes+2,9);
  np = getprime_32(2,newprimes+4,12);
for(i=0; i<6; i++)
 printf("%d. %d \n", i, newprimes[i]);
}
#endif

/***********************************************************************************
 * SPRNG (c) 2014 by Florida State University                                       *
 *                                                                                  *
 * SPRNG is licensed under a                                                        *
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. *
 *                                                                                  *
 * You should have received a copy of the license along with this                   *
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.           *
 ************************************************************************************/
