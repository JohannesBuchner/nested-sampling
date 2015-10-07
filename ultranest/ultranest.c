#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// This flag allows disabling the progressbar if you are having problems
#define ENABLE_PROGRESSBAR
#ifdef ENABLE_PROGRESSBAR
#include <progressbar.h>
#endif

#include "ultranest.h"
#include "draw.h"
#include "sampler.h"

void progressbar_settext(char * buf, int iter, int pbar_maxval, int nlive_points, int ndraws, 
	double logZ, double remainderZ, double logZerr, double remainderZerr, 
	point * current, int ndim)
{
	int i;
	int n = sprintf(buf, "|%d/%d samples+%d/%d|lnZ = %.2f +- %.3f + %.3f|L=%.2e @ ",
		iter + 1, pbar_maxval, nlive_points, ndraws, logaddexp(logZ, remainderZ), logZerr, remainderZerr, current->L);
	for(i = 0; i < ndim && n < 200; i++) {
		n += sprintf(buf + n, "%.3e ", current->phys_coords[i]);
	}
}

void write_results(const char * root, const ultranest_results res, unsigned int ndim) {
	char filename[1000];
	sprintf(filename, "%sposterior_samples.txt", root);
	FILE * fout = fopen(filename, "w");
	if (fout == NULL) {
		perror("ERROR: could not open samples file for writing");
		exit(1);
	}
	for(unsigned int i = 0; i < res.niter; i++) {
		for (unsigned int j = 0; j < ndim; j++)
			fprintf(fout, "%.30e ", res.weighted_points[i].p->phys_coords[j]);
		if (0 == fprintf(fout, "%f %f\n", res.weighted_points[i].p->L, res.weighted_points[i].weight)) {
			perror("ERROR: could not write out samples file");
			exit(1);
		}
	}
	if (fclose(fout) != 0) {
		perror("ERROR: could not close samples file");
		exit(1);
	}
	sprintf(filename, "%sevidence.txt", root);
	fout = fopen(filename, "w");
	if (fout == NULL) {
		perror("ERROR: could not open evidence results file for writing");
		exit(1);
	}
	fprintf(fout, "%f # Evidence\n", res.logZ);
	fprintf(fout, "%f # Evidence Uncertainty\n", res.logZerr);
	fprintf(fout, "%d # Number of iterations\n", res.niter);
	fprintf(fout, "%d # Number of likelihood evaluations (in last run)\n", res.ndraws);
	if (fclose(fout) != 0) {
		perror("ERROR: could not close evidence results file");
		exit(1);
	}
}

ultranest_results ultranest(LikelihoodFunc,
	const char * root, const int ndim, const int max_samples, const double logZtol,
	const int nlive_points, unsigned int nsteps)
{
	unsigned int i = 0;
	double tolerance = logZtol;
	int pbar_maxval = nlive_points;
	char pbar_label[200];
	pbar_label[0] = 0;
	
	ultranest_draw_state * drawer = ultranest_draw_init(Like, ndim, nsteps, 1.);
	#ifdef ENABLE_PROGRESSBAR
	progressbar * pbar = progressbar_new("initialising...", pbar_maxval);
	pbar->format[1] = '=';
	#endif
	ultranest_state * sampler = ultranest_sampler_init(Like, root, ndim, nlive_points, drawer);
	#ifdef ENABLE_PROGRESSBAR
	progressbar_update_label(pbar, "sampling...");
	progressbar_update(pbar, i);
	#endif
	point * current = ultranest_sampler_next(sampler);

	/* begin integration */
	double logVolremaining = 0;
	double logwidth = log(1 - exp(-1. / sampler->nlive_points));
	
	weighted_point * weights = NULL;
	ultranest_results res;
	res.root = root;
	double wi = logwidth + current->L;
	double logZ = wi;
	double H = current->L - logZ;
	double logZerr;
	
	while(1) {
		logwidth = log(1 - exp(-1. / sampler->nlive_points)) + logVolremaining;
		logVolremaining -= 1. / sampler->nlive_points;
		
		weights = (weighted_point *) realloc(weights, (i+sampler->nlive_points+1) * sizeof(weighted_point));
		weights[i].p = current;
		weights[i].weight = logwidth;
		
		i = i + 1;
		logZerr = sqrt(H / sampler->nlive_points);
		
		// double i_final = -sampler->nlive_points * (-sampler->Lmax + logsubexp(fmax(tolerance - logZerr, logZerr / 100.) + logZ, logZ));
		// i_final = -sampler.nlive_points * (-sampler.Lmax + log(exp(max(tolerance - logZerr, logZerr / 100.) + logZ) - exp(logZ)))
		assert(fmax(tolerance - logZerr, logZerr / 100.) > 0);
		assert(fmax(tolerance - logZerr, logZerr / 100.) + logZ > logZ);
		int i_final = -(sampler->nlive_points * (-sampler->Lmax + logsubexp(logZ, fmax(tolerance - logZerr, logZerr / 100.))));
		pbar_maxval = (int) fmin(fmax(i+1, i_final), i+100000);
		
		ultranest_sampler_integrate_remainder(sampler, logwidth, logVolremaining, logZ, NULL);
		
		progressbar_settext(pbar_label, i, pbar_maxval, sampler->nlive_points, sampler->ndraws, 
			logZ, sampler->remainderZ, logZerr, sampler->remainderZerr, current, sampler->ndim);
		#ifdef ENABLE_PROGRESSBAR
		progressbar_update_label(pbar, pbar_label);
		pbar->max = pbar_maxval;
		progressbar_update(pbar, i);
		#else
		printf("%s\n", pbar_label);
		#endif
		
		if (i > sampler->nlive_points) {
			// tolerance
			double total_error = logZerr + sampler->remainderZerr;
			if (max_samples > 0 && (unsigned) max_samples < sampler->ndraws) {
				#ifdef ENABLE_PROGRESSBAR
				progressbar_finish(pbar);
				#endif
				printf("maximum number of samples reached\n");
				break;
			}
			if (total_error < tolerance) {
				#ifdef ENABLE_PROGRESSBAR
				progressbar_finish(pbar);
				#endif
				printf("tolerance reached\n");
				break;
			}
			// we want to make maxContribution as small as possible
			//  but if it becomes 10% of logZerr, that is enough
			if (sampler->remainderZerr < logZerr / 10.) {
				#ifdef ENABLE_PROGRESSBAR
				progressbar_finish(pbar);
				#endif
				printf("tolerance will not improve: remainder error (%.3f) is much smaller than systematic errors (%.3f)\n", sampler->remainderZerr, logZerr);
				break;
			}
		}
		
		current = ultranest_sampler_next(sampler);
		wi = logwidth + current->L;
		double logZnew = logaddexp(logZ, wi);
		H = exp(wi - logZnew) * current->L + exp(logZ - logZnew) * (H + logZ) - logZnew;
		logZ = logZnew;
	}
	// not needed for integral, but for posterior samples, otherwise there
	// is a hole in the most likely parameter ranges.
	i += ultranest_sampler_integrate_remainder(sampler, logwidth, logVolremaining, logZ, weights + i);
	logZerr += sampler->remainderZerr;
	logZ = logaddexp(logZ, sampler->remainderZ);
	
	res.logZ = logZ;
	res.logZerr = logZerr;
	res.ndraws = sampler->ndraws;
	res.niter = i;
	res.H = H;
	res.weighted_points = weights;
	
	printf("ULTRANEST result: lnZ = %.2f +- %.2f\n", res.logZ, res.logZerr);
	write_results(root, res, ndim);
	
	return res;
}




