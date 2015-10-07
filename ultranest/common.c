#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

point * create_point(const unsigned int ndim) {
	point * newpoint = (point *) malloc(sizeof(point));
	newpoint->coords = (double *) calloc(ndim, sizeof(double));
	newpoint->phys_coords = (double *) calloc(ndim, sizeof(double));
	for(unsigned i = 0; i < ndim; i++) {
		newpoint->coords[i] = -1;
		newpoint->phys_coords[i] = -1;
	}
	newpoint->L = 0./0.;
	return newpoint;
}

void free_point(point * p) {
	free(p->coords);
	free(p->phys_coords);
	free(p);
}

double logaddexp(const double a, const double b) {
	/* values between 0 and 1 have highest resolution */
	if (b > a) {
		return log(1 + exp(a - b)) + b;
	} else {
		return log(1 + exp(b - a)) + a;
	}
}
double logsubexp(const double base, const double addition) {
	/* values between 0 and 1 have highest resolution */
	return log(exp(addition) - 1) + base;
}

double compute_distance(const unsigned int ndim, const point * member, const point * other) {
	// compute distance to each member in each dimension
	double distsq = 0;
	for(unsigned int j = 0; j < ndim; j++) {
		distsq += pow(member->coords[j] - other->coords[j], 2);
	}
	return sqrt(distsq);
}

double nearest_rdistance_guess(const unsigned int ndim, const point ** live_points, const unsigned int nlive_points) {
	// determine max distance.
	// using jackknife, is fast: n^2 x ndim
	double maxdistance = 0;
	assert(nlive_points > 0);
	for(unsigned int i = 0; i < nlive_points; i++) {
		// leave ith point out
		double mindistance = 1e300;
		const point * nonmember = live_points[i];
		for (unsigned int k = 0; k < nlive_points; k++) {
			if (k == i)
				continue;
			double dist = compute_distance(ndim, live_points[k], nonmember);
			if (k == 0 || dist < mindistance)
				mindistance = dist;
		}
		// now we have the mindistance. 
		// Use the largest mindistance as maxdistance
		maxdistance = fmax(mindistance, maxdistance);
	}
	return maxdistance;
}



