#include<assert.h>
#include<math.h>
#include<stdio.h>
#include "ultranest.h"

/**
 * \file
 * \mainpage UltraNest
 * 
 * UltraNest uses a hybrid RadFriends/MCMC method for exploring the parameter
 * space. See the accompaning paper for details.
 * 
 * RadFriends is a method to sample from the neighbourhood of existing live 
 * points. This is highly efficient in the beginning of the integration, 
 * and for low-dimensional, peculiar parameter spaces. It also guarantees 
 * perfect uniform sampling.
 * Once this method becomes inefficient, the algorithm switches to MCMC. The 
 * number of steps determines the efficiency 
 * (e.g. accept point after 100 steps = 1% efficiency).
 * Still, RadFriends is used to narrow the MCMC proposal.
 * 
 * Call \ref ultranest to run ultranest.
 * 
 * Download the code from the repository at https://github.com/JohannesBuchner/nested-sampling
 * 
 * \author Johannes Buchner
 * \date 2014
 **/

/**
 * Example Likelihood function: small single-peaked gaussian
 */
void Like1(double * Cube, int * ndim, int * nparams, double * lnew, void * userparams) {
	double l = 0;
	int i;
	double sigma = 0.001;
	/*Cube[0] = Cube[0] * 2;
	Cube[1] = Cube[1] * 2 - 1;*/
	for (i = 0; i < *ndim; i++) {
		l -= 0.5 * pow((Cube[i] - 0.5)/sigma, 2);
	}
	*lnew = l - (*ndim) * (log(sigma) + 0.5 * log(3.14 * 2));
}

/**
 * Example Likelihood function: combination of two gaussians
 */
void Like2(double * Cube, int * ndim, int * nparams, double * lnew, void * userparams) {
	double l1 = 0;
	double l2 = 0;
	int i;
	double sigma = 0.001;
	/*Cube[0] = Cube[0] * 2;
	Cube[1] = Cube[1] * 2 - 1;*/
	for (i = 0; i < *ndim; i++) {
		l1 -= 0.5 * pow((Cube[i] - 0.3)/sigma, 2);
		l2 -= 0.5 * pow((Cube[i] - 0.6)/sigma, 2);
	}
	*lnew = logaddexp(l1, l2) - (*ndim) * (log(sigma) + 0.5 * log(3.14 * 2)) - log(2);
}

/**
 * Example program that shows how to use ultranest.
 */
int main(void) {
	char * prefix = "example";
	unsigned int ndim = 10;
	ultranest_results res = ultranest(Like2, prefix, ndim, -1, 0.1, 1000, 50);
	return 0;
}

