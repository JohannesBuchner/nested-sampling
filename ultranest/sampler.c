#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"
#include "sampler.h"
#include "draw.h"

int live_point_cmp(const void * va, const void * vb) {
	const point ** ppa = (const point **) va;
	const point ** ppb = (const point **) vb;
	const point * pa = *ppa;
	const point * pb = *ppb;
	
	if (pa->L < pb->L)
		return -1;
	if (pa->L > pb->L)
		return 1;
	fprintf(stderr, "warning: live points with exactly same likelihood: %e\n", pa->L);
	return 0;
}

int draw_point_cmp(const void * va, const void * vb) {
	const draw_point * pa = (const draw_point *) va;
	const draw_point * pb = (const draw_point *) vb;
	
	if (pa->Lmin < pb->Lmin) return 1; // reverse, i.e. highest Lmin first
	if (pa->Lmin > pb->Lmin) return -1;
	return 0;
}

int sort_live_points(ultranest_state * state) {
	qsort(state->live_points, state->nlive_points, sizeof(point *), live_point_cmp);
	/*printf("sorted live points:\n");
	for(unsigned int i = 0; i < state->nlive_points; i++) {
		printf("  %d: %f\n", i, state->live_points[i]->L);
	}*/
	return 0;
}

void write_draw(FILE * f, draw_point * dp, unsigned int ndim) {
	int r;
	unsigned int j;
	r = fprintf(f, "%.20e %.20e ", dp->Lmin, dp->p->L);
	if (r <= 0) {
		perror("ERROR: could not write out current draws correctly!");
		exit(1);
	}
	for (j = 0; j < ndim; j++) {
		r = fprintf(f, "%.20e ", dp->p->coords[j]);
		if (r <= 0) {
			perror("ERROR: could not write out current draws correctly!");
			exit(1);
		}
	}
	for (j = 0; j < ndim; j++) {
		r = fprintf(f, "%.20e ", dp->p->phys_coords[j]);
		if (r <= 0) {
			perror("ERROR: could not write out current draws correctly!");
			exit(1);
		}
	}
	fprintf(f, "\n");
}
void write_draw_(FILE * f, double Lmin, point * p, unsigned int ndim) {
	draw_point dp;
	dp.p = p;
	dp.Lmin = Lmin;
	write_draw(f, &dp, ndim);
}

int read_draw(FILE * f, draw_point * dp, unsigned int ndim) {
	int r;
	unsigned int j;
	r = fscanf(f, "%lf %lf", &(dp->Lmin), &(dp->p->L));
	if (r == EOF) return 0;
	if (r != 2) {
		perror("ERROR: could not read previous draws correctly (Lmin, L)!");
		exit(1);
	}
	for (j = 0; j < ndim; j++) {
		r = fscanf(f, "%lf", &(dp->p->coords[j]));
		if (r != 1) {
			perror("ERROR: could not read previous draws correctly (coords)!");
			exit(1);
		}
	}
	for (j = 0; j < ndim; j++) {
		r = fscanf(f, "%lf", &(dp->p->phys_coords[j]));
		if (r != 1) {
			perror("ERROR: could not read previous draws correctly (physcoords)!");
			exit(1);
		}
	}
	return 1;
}

void write_checkpoint(const ultranest_state * state) {
	char filename[1024];
	char tmpfile[1024];
	sprintf(filename, "%srng.dat", state->root);
	sprintf(tmpfile, "%srng_next.dat", state->root);
	
	FILE * f = fopen(tmpfile, "w");
	if (f == NULL) {
		perror("ERROR: could not open rng state file for writing");
		exit(1);
	}
	if (gsl_rng_fwrite(f, state->draw_state->rng) != 0) {
		perror("ERROR: could not write rng state");
		exit(1);
	}
	if (fclose(f) != 0) {
		perror("ERROR: could not close rng state file");
		exit(1);
	}

	if (rename(tmpfile, filename) != 0) {
		perror("ERROR: could not overwrite rng state file by renaming temp file!");
		exit(1);
	}
	// rng should be ahead of draws
	fflush(state->draws_file);
}
int load_checkpoint(ultranest_state * state) {
	char filename[1024];
	sprintf(filename, "%srng.dat", state->root);
	FILE * f = fopen(filename, "r");
	if (f != NULL) {
		if (gsl_rng_fread(f, state->draw_state->rng) != 0) {
			perror("ERROR: could not read rng state");
			exit(1);
		}
		if (fclose(f) != 0) {
			perror("ERROR: could not close rng state file");
			exit(1);
		}
		return 0;
	}
	return 1;
}

void ultranest_sampler_load_previous_draws(ultranest_state * state) {
	// load table of previous draws for re-use/continuation
	unsigned int i = 0;
	unsigned int r;
	char filename[1024];
	sprintf(filename, "%sdraws.txt", state->root);
	FILE * f = fopen(filename, "r");
	
	state->draws_prev = NULL;
	state->draws_prev_n = 0;
	
	if (f != NULL) {
		printf("Loading draws file..."); fflush(stdout);
		while (1) {
			// try to load a line
			// make space to store
			state->draws_prev = (draw_point *) realloc(state->draws_prev, (i+1) * sizeof(draw_point));
			if(state->draws_prev == NULL) {
				perror("ERROR: could not allocate more memory for loading previous draws!");
				exit(1);
			}
			state->draws_prev[i].p = create_point(state->ndim);
			// is this the end of the file?
			if (feof(f)) {
				r = fclose(f);
				if (r != 0) {
					perror("ERROR: could not read previous draws correctly: closing file failed!");
					exit(1);
				}
				break;
			}
			
			r = read_draw(f, &state->draws_prev[i], state->ndim);
			if (r == 0) { // EOF
				break;
			}
			// successful so far (otherwise we would have crashed).
			// set new size
			state->draws_prev_n = i + 1;
			i++;
		}
		printf("done.\n"); fflush(stdout);
	}
	if (state->draws_prev_n > 0)
		printf("Loaded %d draws from previous run\n", state->draws_prev_n);
	
	// now we have 0 or more draws. Sort them by decreasing Lmin
	// so that we can pop them from the end.
	qsort(state->draws_prev, state->draws_prev_n, sizeof(draw_point), draw_point_cmp);

	if (state->draws_prev_n > 0 && load_checkpoint(state) != 0) {
		fprintf(stderr, "ERROR: could not load rng state, but only previous samples.\n"
			"This is dangerous, because we will re-draw the exact same points.\n"
			"You can solve this problem by deleting the '%s' file\n", filename);
		exit(1);
	}
	
	// immediately write out these draws.
	// because if we crash, we should not lose our data. If we sample, it 
	// just continuously adds to the data.
	
	char tmpfile[1024];
	sprintf(tmpfile, "%sdraws_next.txt", state->root);
	state->draws_file = fopen(tmpfile, "w");
	if (state->draws_file == NULL) {
		perror("ERROR: could not open draws file for writing!");
		exit(1);
	}
	for (i = 0; i < state->draws_prev_n; i++) {
		write_draw(state->draws_file, &state->draws_prev[i], state->ndim);
	}
	if (rename(tmpfile, filename) != 0) {
		perror("ERROR: could not overwrite draws file by renaming temp file!");
		exit(1);
	}
	// flush to disk (only here)
	fflush(state->draws_file);
}

int replace_from_cache(ultranest_state * state, double Lmin, point * replacement) {
	//  check cache
	while (state->draws_prev_n > 0) {
		// go to last item
		int k = state->draws_prev_n - 1;
		// is suitable if Lmin is lower than what is needed (superset)
		// and L is higher (target zone)
		
		// if too far ahead, stop.
		if (Lmin < state->draws_prev[k].Lmin) {
			// can not sample from this subset. Need outer superset.
			break;
		}
		
		// regardless of whether this point was useful or not, 
		// it is not useful for next time. So remove by shrinking.
		state->draws_prev_n -= 1;
		if (state->draws_prev[k].Lmin <= Lmin &&
			state->draws_prev[k].p->L > Lmin) {
			
			// copy over
			replacement->L = state->draws_prev[k].p->L;
			for(unsigned int j = 0; j < state->ndim; j++) {
				replacement->coords[j] = state->draws_prev[k].p->coords[j];
				replacement->phys_coords[j] = state->draws_prev[k].p->phys_coords[j];
			}
			// success
			return 1;
		}
		// save some space: 
		free(state->draws_prev[k].p);
		state->draws_prev = (draw_point *) realloc(state->draws_prev, (state->draws_prev_n) * sizeof(draw_point));
	}
	// nothing found
	return 0;
}

ultranest_state * ultranest_sampler_init(LikelihoodFunc, 
	const char * root, int ndim, unsigned int nlive_points, 
	ultranest_draw_state * draw_state) 
{
	ultranest_state * state = (ultranest_state *) malloc(sizeof(ultranest_state));
	assert(ndim > 0);
	state->root = root;
	state->nlive_points = nlive_points;
	state->remainderZ = 0./0.;
	state->ndraws = 0;
	state->Lmax = 0./0.;
	state->ndim = ndim;
	state->Like = Like;
	state->live_points = (point **) calloc(nlive_points, sizeof(point *));
	state->draw_state = draw_state;
	ultranest_sampler_load_previous_draws(state);
	
	/* sample initial live points */
	for (unsigned int i = 0; i < nlive_points; i++) {
		point * current = create_point(ndim);
		double Lmin = -1e300;
		current->L = Lmin;
		
		int from_cache = replace_from_cache(state, Lmin, current);
		if (from_cache == 0) {
			ultranest_draw_generate_direct(state->draw_state, current, NULL, 0);
			state->Like(current->phys_coords, &ndim, &ndim, &current->L, NULL);
			state->ndraws += 1;
			write_draw_(state->draws_file, Lmin, current, state->ndim);
			write_checkpoint(state);
		}
		
		if (i == 0)
			state->Lmax = current->L;
		else
			state->Lmax = fmax(state->Lmax, current->L);
		state->live_points[i] = current;
	}
	sort_live_points(state);
	return state;
}

point * ultranest_sampler_next(ultranest_state * state) {
	// select worst point
	int i = 0;
	unsigned int j;
	point * current = state->live_points[i];
	// make a copy
	point * replacement = create_point(state->ndim);
	replacement->L = current->L;
	for(j = 0; j < state->ndim; j++) {
		replacement->coords[j] = current->coords[j];
		replacement->phys_coords[j] = current->phys_coords[j];
	}
	// replace point
	
	unsigned int ndraws = 0;
	int from_cache = replace_from_cache(state, current->L, replacement);
	
	if (from_cache == 0) {
		ndraws += ultranest_draw_next(state->draw_state, replacement, (const point **)state->live_points, (const unsigned int)state->nlive_points);
		write_draw_(state->draws_file, current->L, replacement, state->ndim);
		write_checkpoint(state);
	}
	
	state->live_points[i] = replacement;
	// keep points sorted
	sort_live_points(state);
	
	if (replacement->L > state->Lmax) {
		printf("\nNew MaxLike: %.2e: @ ", replacement->L);
		for(j = 0; j < state->ndim; j++) {
			printf("%.3e ", replacement->phys_coords[j]);
		}
		printf("\n");
		state->Lmax = replacement->L;
	}
	state->ndraws += ndraws;
	// return removed point
	return current;
}

/* if points is not NULL, copy the points there; returns number of items copied */
int ultranest_sampler_integrate_remainder(ultranest_state * state, 
	const double logwidth, const double logVolremaining, const double logZ, 
	weighted_point * points
) {
	// logwidth remains the same now for each sample
	unsigned int i;
	unsigned int n = state->nlive_points;
#if(0)
	double maxContribution = state->Lmax + logVolremaining;
	double logZup  = logaddexp(maxContribution, logZ);
	double remainderZ = maxContribution;
	double remainderZerr = logZup - logZ;
#else
	// more careful assessment:
	double logV = logwidth;
	double L0 = state->live_points[state->nlive_points - 1]->L;
	
	double Lmax = 0;
	double Lmin = 0;
	double Lmid = 0;
	// the positive edge is L2, L3, ... L-1, L-1
	// the average  edge is L1, L2, ... L-2, L-1
	// the negative edge is L1, L1, ... L-2, L-2

	for(i = 0; i < n; i++) {
		double Ldiff = exp(state->live_points[i]->L - L0);
		
		if (i > 0)
			Lmax += Ldiff;
		if (i == n - 1)
			Lmax += Ldiff;
		if (i < n - 1)
			Lmin += Ldiff;
		if (i == 0)
			Lmin += Ldiff;
		Lmid += Ldiff;
	}
	//Lmax = Ls[1:].sum() + Ls[-1]
	//Lmin = Ls[:-1].sum() + Ls[0]
	
	assert(Lmax >= Lmin);
	assert(Lmax >= Lmid);
	assert(Lmax >= Lmin);
	
	double logZmid = logaddexp(logZ, logV + log(Lmid) + L0);
	double logZup  = logaddexp(logZ, logV + log(Lmax) + L0);
	double logZlo  = logaddexp(logZ, logV + log(Lmin) + L0);
	double logZerr = fmax(logZup - logZmid, logZmid - logZlo);
	assert(logZup > logZmid);
	assert(logZmid > logZlo);
	double remainderZ = logV + log(Lmid) + L0;
	double remainderZerr = logZerr;
#endif
	assert(remainderZerr > 0);
	/* update remainderZ, remainderZerr in state */
	state->remainderZ = remainderZ;
	state->remainderZerr = remainderZerr;
	for(i = 0; i < n; i++) {
		if (points != NULL) {
			points[i].weight = logwidth;
			points[i].p = state->live_points[i];
		}
	}
	return n;
}



