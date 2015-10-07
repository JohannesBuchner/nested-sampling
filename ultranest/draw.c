#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include "common.h"
#include "draw.h"

gsl_rng * init_rng() {
	const gsl_rng_type * T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	return gsl_rng_alloc(T);
}

ultranest_draw_state * ultranest_draw_init(LikelihoodFunc, unsigned int ndim, unsigned int nsteps, double proposal_scale) 
{
	ultranest_draw_state * state = (ultranest_draw_state *) malloc(sizeof(ultranest_draw_state));
	
	state->Like = Like;
	state->ndim = ndim;
	state->rng = init_rng();
	state->phase = 0;
	state->niter = 0;
	
	state->ndirect = 0;
	state->ndirect_accepts = 0;
	state->region_low = (double *) calloc(ndim, sizeof(double));
	state->region_high = (double *) calloc(ndim, sizeof(double));
	for (unsigned int i = 0; i < ndim; i++) {
		state->region_low[i] = 0;
		state->region_high[i] = 1;
	}
	
	state->nfriends = 0;
	state->nfriends_accepts = 0;
	state->maxdistance = -1;
	
	state->nsteps = nsteps;
	state->proposal_scale = proposal_scale;
	state->mcmc_naccepts = 0;
	state->mcmc_nrejects = 0;
	state->mcmc_naccepts_total = 0;
	state->mcmc_nrejects_total = 0;
	state->mcmc_nskip = 0;
	return state;
}

unsigned int ultranest_draw_is_inside(const ultranest_draw_state * state, const point * current, const point ** members, const unsigned int nmembers) {
	// Check if this new point is near or inside one of our clusters
	// return 0 if not inside, and the number of nearby members otherwise
	unsigned int ndim = state->ndim;
	unsigned int i;
	
	// if it does not even lie in our primitive rectangle,
	//   do not even need to compute the distances
	for (i = 0; i < ndim; i++) {
		if (current->coords[i] < state->region_low[i])
			return 0;
		if (current->coords[i] > state->region_high[i])
			return 0;
	}
	
	double maxdistance = state->maxdistance;
	
	// if not initialized: no prefiltering
	if (!(maxdistance > 0))
		return 1;
	
	unsigned int nnearby = 0;
	// compute distance to each member in each dimension
	for(i = 0; i < nmembers; i++) {
		double dist = compute_distance(ndim, members[i], current);
		if (dist <= maxdistance) { // is close to this point
			nnearby += 1;
		}
	}
	return nnearby;
}

unsigned int ultranest_draw_generate_direct(ultranest_draw_state * state, point * current, const point ** members, const unsigned int nmembers) {
	unsigned int ntotal = 0;
	unsigned int ndim = state->ndim;
	
	while(1) {
		// draw directly from prior, within rectangle
		for (unsigned int j = 0; j < ndim; j++) {
			current->coords[j] = gsl_rng_uniform(state->rng) * (
				state->region_high[j] - state->region_low[j]) + state->region_low[j];
			current->phys_coords[j] = current->coords[j];
		}
		ntotal += 1;
		state->ndirect += 1;
		if (nmembers == 0) {
			// can not check region, assume it's ok
			return ntotal;
		}
		if (ultranest_draw_is_inside(state, current, members, nmembers) > 0) {
			// is within region, and ok
			return ntotal;
		}
		if (ntotal > 10000) {
			// this strategy is not working
			return ntotal;
		}
	}
}

unsigned int ultranest_draw_generate_from_friends(ultranest_draw_state * state, point * current, const point ** members, const unsigned int nmembers) {
	// for small regions draw from points
	unsigned int ntotal = 0;
	double maxdistance = state->maxdistance;
	assert(maxdistance > 0);
	unsigned int ndim = state->ndim;
	double direction[ndim];
	unsigned int j;
	
	while(1) {
		// choose random friend
		const point * member = members[gsl_rng_uniform_int(state->rng, nmembers)];
		ntotal += 1;
		// draw direction around it
		double lengthsq = 0;
		for(j = 0; j < ndim; j++) {
			direction[j] = gsl_ran_gaussian(state->rng, 1);
			lengthsq += pow(direction[j], 2);
		}
		double length = sqrt(lengthsq);
		for(j = 0; j < ndim; j++) {
			direction[j] /= length;
		}
		// choose radius: volume gets larger towards the outside
		// so give the correct weight with dimensionality
		double radius = maxdistance * pow(gsl_rng_uniform(state->rng), 1./ndim);
		for(j = 0; j < ndim; j++) {
			current->coords[j] = member->coords[j] + direction[j] * radius;
			current->phys_coords[j] = current->coords[j];
		}
		// count the number of points this is close to
		int nnearby = ultranest_draw_is_inside(state, current, members, nmembers);

		if (ntotal > 10000) {
			// this strategy is not working
			return ntotal;
		}

		if (nnearby == 0) {
			// Point should lie inside by construction
			// is outside region. Try again.
			// should never happen, except for rounding issues
			continue;
		}
		// accept with probability 1./nnear
		double coin = gsl_rng_uniform(state->rng);
		if (coin < 1. / nnearby) { // accept
			return ntotal;
		}
	}
}

unsigned int ultranest_draw_local_step_sample(ultranest_draw_state * state, 
	point * current, double Lmin, const point ** members, const int nmembers) {

	assert(current->L >= Lmin);
	unsigned int k = 0;
	for (unsigned int step = 0; step < state->nsteps; step++) {
	
		double scale = state->maxdistance * state->proposal_scale;
		// restricted proposal
		int ndim = state->ndim;
		point * proposal = create_point(state->ndim);
		unsigned int j;
	
		for (unsigned int i = 0; i < 1000; i++) {
			for(j = 0; j < state->ndim; j++) {
				proposal->coords[j] = current->coords[j] + gsl_ran_gaussian(state->rng, scale);
				proposal->phys_coords[j] = proposal->coords[j];
			}
			if (ultranest_draw_is_inside(state, current, members, nmembers) != 0) {
				// is inside superset, so perhaps good.
				break;
			} else {
				// can be rejected!
				state->mcmc_nskip += 1;
			}
		}
		// check if actually inside
		proposal->L = Lmin;
		state->Like(proposal->phys_coords, &ndim, &ndim, &proposal->L, NULL);
	
		k += 1;
		if (proposal->L >= Lmin) {
			// is ok, accept the proposed point by copying it over.
			for(j = 0; j < state->ndim; j++) {
				current->coords[j] = proposal->coords[j];
				current->phys_coords[j] = proposal->phys_coords[j];
			}
			current->L = proposal->L;
			state->mcmc_naccepts += 1;
		} else {
			// otherwise, remain at the current point.
			state->mcmc_nrejects += 1;
		}
	
		free_point(proposal);
	}
	return k;
}

void copy_random_livepoint(ultranest_draw_state * state, point * current, const point ** members, const int nmembers) {
	unsigned int i = gsl_rng_uniform_int(state->rng, nmembers);
	// copy start point over
	for (unsigned int j = 0; j < state->ndim; j++) {
		current->coords[j] = members[i]->coords[j];
		current->phys_coords[j] = members[i]->phys_coords[j];
	}
	current->L = members[i]->L;
}

void ultranest_draw_adapt(ultranest_draw_state * state) {
	unsigned int n = state->mcmc_naccepts + state->mcmc_nrejects;
	double proposal_scale = state->proposal_scale;
	double ar = state->mcmc_naccepts * 100. / n;
	
	if (ar > 0.5) {
		state->proposal_scale *= exp(1./state->mcmc_naccepts);
	} else {
		state->proposal_scale /= exp(1./state->mcmc_nrejects);
	}
	
	if (ar < 25) {
		printf("\nproposal scale %.2f, acceptance rate: %3.2f%% %50s\n", proposal_scale, ar, "v");
	} else if (ar > 75) {
		printf("\nproposal scale %.2f, acceptance rate: %3.2f%% %50s\n", proposal_scale, ar, "^");
	}
	state->mcmc_naccepts_total += state->mcmc_naccepts;
	state->mcmc_nrejects_total += state->mcmc_nrejects;
	state->mcmc_naccepts = 0;
	state->mcmc_nrejects = 0;
}

unsigned int ultranest_draw_local_step_calibrate(ultranest_draw_state * state, 
	point * current, double Lmin, const point ** members, const int nmembers) {
	unsigned int k = 0;
	unsigned int prev_good = 0;
	
	// make sure we get some accepts and some rejects
	while(1) {
		state->mcmc_naccepts = 0;
		state->mcmc_nrejects = 0;
		copy_random_livepoint(state, current, members, nmembers);
		k += ultranest_draw_local_step_sample(state, current, Lmin, members, nmembers);
		
		double ar = state->mcmc_naccepts * 100. / (state->mcmc_naccepts+state->mcmc_nrejects);
		
		if (ar < 10 || state->mcmc_naccepts < 2) {
			// never leaving. Decrease scale drastically
			printf("Calibration: Only %d of %d steps accepted (%.3f%% AR) --> Scaling down proposal (%.4f --> %.4f)\n", 
				state->mcmc_naccepts, state->mcmc_naccepts+state->mcmc_nrejects, 
					ar, state->proposal_scale, state->proposal_scale / 2);
			state->proposal_scale /= 2;
			prev_good = 0;
		} else if (ar > 90 || state->mcmc_nrejects < 2) {
			// never hitting border. Increase scale drastically
			printf("Calibration: Only %d of %d steps rejected (%.3f%% AR) --> Scaling up   proposal (%.4f --> %.4f)\n", 
				state->mcmc_nrejects, state->mcmc_naccepts+state->mcmc_nrejects, 
					ar, state->proposal_scale, state->proposal_scale * 2);
			state->proposal_scale *= 2;
			prev_good = 0;
		} else {
			ultranest_draw_adapt(state);
			if (prev_good == 1) {
				// stop when we hit 2 good ones in a row.
				break;
			}
			prev_good = 1;
		}
	}
	return k;
}



void ultranest_draw_print_stats(const ultranest_draw_state * state) {
	unsigned int n = state->mcmc_naccepts_total + state->mcmc_nrejects_total;
	if (n == 0) return;
	double ar = state->mcmc_naccepts_total * 100. / n;
	double boost = state->mcmc_nskip * 100. / n;
	printf("MCMCConstrainer stats: nsteps: %d, %.3f%% accepts, %.3f%% skipped (%d)",
		n, ar, boost, state->mcmc_nskip);
}



unsigned int ultranest_draw_next(ultranest_draw_state * state, point * current, 
	const point ** live_points, const unsigned int nlive_points)
{
	state->niter += 1;
	double Lmin = current->L;
	int ndim = state->ndim;
	// compute RadFriends spheres
	if (!(state->maxdistance > 0) || state->niter % 50 == 1) {
		double maxdistance = nearest_rdistance_guess(state->ndim, live_points, nlive_points);
		// make sure we only shrink
		if (state->maxdistance > 0 && state->maxdistance < maxdistance)
			maxdistance = state->maxdistance;
		
		// compute enclosing rectangle for quick checks
		for (unsigned int j = 0; j < state->ndim; j++) {
			double low = 1, high = 0;
			for (unsigned int i = 0; i < nlive_points; i++) {
				low = fmin(low, live_points[i]->coords[j]);
				high = fmax(high, live_points[i]->coords[j]);
				assert(low <= high);
			}
			state->region_low[j] = fmax(0, low - maxdistance);
			state->region_high[j] = fmin(1, high + maxdistance);
			assert(state->region_low[j] < state->region_high[j]);
			assert(state->region_low[j] >= 0);
			assert(state->region_high[j] <= 1);
		}
		state->maxdistance = maxdistance;
	}
		
	unsigned int ntoaccept = 0;
	if (state->phase == 0) {
		// draw from rectangle until sphere rejection drops below 1%
		while(1) {
			unsigned int ntotal = ultranest_draw_generate_direct(state, current, live_points, nlive_points);
			// reset, in case it was overwritten
			current->L = Lmin;
			state->Like(current->phys_coords, &ndim, &ndim, &current->L, NULL);
			
			state->nfriends += 1;
			ntoaccept += 1;
			if (current->L >= Lmin) {
				if (ntoaccept >= 200)
					printf("\nDirect sampling: RadFriends is becoming inefficient: %d draws until accept (switching at 400) \n", ntoaccept);
				state->nfriends_accepts += 1;
				return ntoaccept;
			}
			if (ntotal >= 20) {
				// drawing directly from prior becomes 
				// inefficient as we go to small region.
				// switch to drawing from RadFriends
				printf("\nswitching to RadFriends sampling phase\n");
				state->phase = 1;
				break;
			}
			if (ntoaccept >= 400 && state->nsteps > 0) {
				// drawing using RadFriends can become
				// inefficient in high dimensionality
				// switch to local step sampling
				printf("\nswitching to local steps sampling phase\n");
				state->phase = 2;
				break;
			}
		}
	}
	if(state->phase == 1) {
		// draw from spheres until acceptance rate drops below 0.05%
		while (1) {
			/*unsigned int ntotal =*/ ultranest_draw_generate_from_friends(state, current, live_points, nlive_points);
			// reset, in case it was overwritten
			current->L = Lmin;
			state->Like(current->phys_coords, &ndim, &ndim, &current->L, NULL);
			state->nfriends += 1;
			ntoaccept += 1;
			if (current->L >= Lmin) {
				if (ntoaccept >= 200)
					printf("\nRadFriends sampling: RadFriends is becoming inefficient: %d draws until accept  (switching at 400) \n", ntoaccept);
				state->nfriends_accepts += 1;
				return ntoaccept;
			}
			if (ntoaccept >= 400 && state->nsteps > 0) {
				// drawing using RadFriends can become
				// inefficient in high dimensionality
				// switch to local step sampling
				printf("\nswitching to local steps sampling phase\n");
				state->phase = 2;
				break;
			}
		}
	}
	unsigned int k = 0;
	// everything else failed, so do local step sampling
	// start from random point, not necessarily the one being removed
	if (state->mcmc_naccepts_total == 0) {
		// first time for entering mcmc. Need to calibrate first.
		printf("\nCalibrating proposal...\n");
		k += ultranest_draw_local_step_calibrate(state, current, Lmin, live_points, nlive_points);
		printf("Calibrating proposal complete.\n");
	}
	
	copy_random_livepoint(state, current, live_points, nlive_points);
	k += ultranest_draw_local_step_sample(state, current, Lmin, live_points, nlive_points);
	ultranest_draw_adapt(state);
	
	return k + ntoaccept;
}

