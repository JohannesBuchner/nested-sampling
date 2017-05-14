/***
This file is part of nested_sampling, a pythonic implementation of various
nested sampling algorithms.

Author: Johannes Buchner (C) 2013-2016
License: AGPLv3

See README and LICENSE file.
***/

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<math.h>
#ifdef PARALLEL
#include<omp.h>
#endif

#define IFVERBOSE if(0)
#define IFDEBUG if(0)
#define adouble double
#define bdouble double
#define sqr(x) (pow(x,2))

double most_distant_nearest_neighbor(
	const void * xxp, int nsamples, int ndim
) {
	const adouble * xx = (const adouble*) xxp;
	double nearest_ds[nsamples];

	IFVERBOSE {
		for (int i = 0; i < nsamples; i++) { // one sample at a time
			printf("%d: ", i);
			for (int k = 0; k < ndim; k++) {
				printf("%e\t", xx[i*ndim + k]);
			}
			printf("\n");
		}
	}
	#ifdef PARALLEL
	#pragma omp parallel for
	#endif
	for (int i = 0; i < nsamples; i++) { // one sample at a time
		// consider all other samples before i
		double nearest_d = 1e300;
		for (int j = 0; j < nsamples; j++) {
			if (j != i) {
				double d = 0;
				for (int k = 0; k < ndim; k++) {
					d += sqr(xx[i*ndim + k] - xx[j*ndim + k]);
				}
				if (d < nearest_d) {
					nearest_d = d;
				}
			}
		}
		IFVERBOSE printf("%d: %f\n", i, sqrt(nearest_d));
		nearest_ds[i] = sqrt(nearest_d);
	}
	double furthest_d = nearest_ds[0];

	for (int i = 1; i < nsamples; i++) {
		if (nearest_ds[i] > furthest_d)
			furthest_d = nearest_ds[i];
	}
	IFVERBOSE printf("result: %f\n", furthest_d);
	return furthest_d;
}

