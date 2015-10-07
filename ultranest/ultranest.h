#ifndef ULTRANEST
#define ULTRANEST

#include "common.h"

/**
 * \file
 * \author Johannes Buchner
 * \date 2014
 * \copyright AGPLv3
 **/

/**
 * UltraNest results
 */
typedef struct {
	/// output prefix used
	const char * root;
	/// natural logarithm of the evidence
	double logZ;
	/// uncertainty on the evidence
	double logZerr;
	/// information
	double H;
	/// number of likelihood evaluations in this run
	unsigned int ndraws;
	/// total number of iterations
	unsigned int niter;
	/// number of dimensions
	unsigned int ndim;
	/// posterior samples with weights
	weighted_point * weighted_points;
} ultranest_results;

/**
 * Run UltraNest.
 *
 * \parblock
 * \param Like
 *    Likelihood function. For example
 *    
 *    \code{.c}
 *    void MyLike(double * Cube, int * ndim, int * nparams, double * lnew, void * userparams) {
 *         // do a transformation of variables
 *         Cube[0] = Cube[0] * 2;
 *         // compute and store likelihood
 *         *lnew = pow((Cube[0] - 0.3) / 0.01, 2);
 *    }
 *    \endcode
 * 
 * \param root  prefix for output files
 * 
 * \param ndim  dimensionality of the problem
 * 
 * \param max_samples  maximum number of likelihood evaluations to perform
 *                     Set to -1 to not set a limit.
 * 
 * \param logZtol  tolerance on the evidence to achieve (e.g. log(10)). Once
 *                 this tolerance is achieved, integration is done. Higher 
 *                 values mean the integration finishes earlier.
 * 
 * \param nlive_points  number of live points to use
 *                      Use 50 for a quick look, 400 or 1000 for a real, 
 *                      typical problem.
 * 
 * \param nsteps  number of MCMC proposal steps
 * 
 * \endparblock
 * 
 * @returns a ultranest_results structure with the results.
 * 
 * Additionally, the results are written to files:
 * 
 * <root>posterior_samples.txt: Contains coordinates, likelihood and weights
 *       as a table.
 * <root>evidence.txt: Contains 4 lines, with one number each: 
 *       evidence, uncertainty, number of iterations and the 
 *       number of likelihood evaluations.
 * 
 * Additional output files are produces so a run can be continued.
 * 
 * Uses \ref ultranest_draw_init, \ref ultranest_sampler_init, \ref ultranest_sampler_next and 
 *   \ref ultranest_sampler_integrate_remainder
 */
ultranest_results ultranest(LikelihoodFunc,
	const char * root, const int ndim, const int max_samples, const double logZtol,
	const int nlive_points, unsigned int nsteps);

#endif
