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

int is_within_distance_of(
	const void * xxp, int nsamples, int ndim, double maxdistance, const void * yp
) {
	const adouble * xx = (const adouble*) xxp;
	const adouble * y = (const adouble*) yp;

	for (int i = 0; i < nsamples; i++) { // one sample at a time
		double d = 0;
		for (int k = 0; k < ndim; k++) {
			d += sqr(xx[i*ndim + k] - y[k]);
		}
		if (sqrt(d) < maxdistance)
			return 1;
	}
	return 0;
}


int count_within_distance_of(
	const void * xxp, int nsamples, int ndim, double maxdistance, 
	const void * yyp, int nothers, void * outp, const int countmax
) {
	const adouble * xx = (const adouble*) xxp;
	const adouble * yy = (const adouble*) yyp;
	double * out = (double*) outp;

	for (int j = 0; j < nothers; j++) { // one sample at a time
		for (int i = 0; i < nsamples; i++) { // one sample at a time
			double d = 0;
			for (int k = 0; k < ndim; k++) {
				d += sqr(xx[i*ndim + k] - yy[j*ndim + k]);
			}
			if (sqrt(d) < maxdistance) {
				out[j]++;
				// printf("%d: %f\n", j, out[j]);
				if (countmax > 0 && out[j] >= countmax) {
					break;
				}
			}
		}
	}
	return 0;
}

const double * current_dists;

int compare_dists(const void * ap, const void * bp) {
	const int a = * (int *) ap;
	const int b = * (int *) bp;
	if (current_dists[a] < current_dists[b]) {
		return -1;
	} else {
		return +1;
	}
}

/**
 *
 * clusters needs to be set to arange(n)
 */

int jarvis_patrick_clustering(
	const void * xxp, int n, int K, int J, 
	int * clusters
) {
	const adouble * dists = (const adouble*) xxp;
	int neighbors_list[n][K];
	int neighbors_list_i[n];
	
	for (int i = 0; i < n; i++) {
		// order its nearest neighbors
		for (int j = 0; j < n; j++) {
			neighbors_list_i[j] = j;
		}
		current_dists = dists + i * n;
		qsort(neighbors_list_i, n, sizeof(int), compare_dists);
		// now neighbors_list_i should be sorted, with nearest at 0
		// we want 1...K+1
		for (int j = 0; j < K; j++) {
			neighbors_list[i][j] = neighbors_list_i[j+1];
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {
			// count how many in common in neighbors_list
			int in_common = 0;
			for (int k = 0; k < K; k++) {
				int a = neighbors_list[i][k];
				for (int k2 = 0; k2 < K; k2++) {
					if (neighbors_list[j][k] == a) {
						in_common++;
						break;
					}
				}
			}
			if (in_common >= J) {
				// re-assign clusters
				int c1 = clusters[i];
				int c2 = clusters[j];
				if (c1 == c2)
					continue;
				if (c1 > c2) {
					c1 = clusters[j];
					c2 = clusters[i];
				}
				// move all from c2 to c1
				for (int k = 0; k < n; k++) {
					if (clusters[k] == c2) {
						clusters[k] = c1;
					}
				}
			}
		}
	}

	return 0;
}

