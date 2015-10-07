#ifndef ULTRANEST_DRAW
#define ULTRANEST_DRAW

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include "common.h"

/**
 * \file
 * \brief Constrained draw method: RadFriends/MCMC hybrid
 */

/**
 * Private variables of the drawing method
 */
typedef struct {
	/// dimensionality
	unsigned int ndim;
	/// PRNG state
	gsl_rng * rng;
	/// Likelihood function
	LikelihoodFunc;
	/// iteration counter
	unsigned int niter;
	/**
	 * current phase:
	 *  0 -- draw directly from cube, reject using RadFriends
	 *  1 -- draw from points using RadFriends
	 *  2 -- draw using MCMC, with RadFriends constrained proposal
	 */
	unsigned int phase;
	/* direct sampling specific */
	/// number of direct draws
	unsigned int ndirect;
	/// number of accepts by RadFriends
	unsigned int ndirect_accepts;
	/// Bounding rectangle on sampling region: low
	double * region_low;
	/// Bounding rectangle on sampling region: high
	double * region_high;
	/* RadFriends specific */
	/// number of RadFriends draws
	unsigned int nfriends;
	/// number of accepts through likelihood constrain
	unsigned int nfriends_accepts;
	/// characteristic distance of RadFriends
	double maxdistance;
	/* MCMC specific */
	/// number of MCMC steps
	unsigned int nsteps;
	/// scale of proposal (multiplied by \ref maxdistance)
	double proposal_scale;
	/// MCMC accepts of current chain
	unsigned int mcmc_naccepts;
	/// MCMC rejects of current chain
	unsigned int mcmc_nrejects;
	/// MCMC skips due to RadFriends
	unsigned int mcmc_nskip;
	/// MCMC accepts (total)
	unsigned int mcmc_naccepts_total;
	/// MCMC rejects (total)
	unsigned int mcmc_nrejects_total;
} ultranest_draw_state;

/**
 * Create the RadFriends/MCMC hybrid drawing method
 * 
 * For use with \ref ultranest_sampler_init.
 *
 * \param proposal_scale Initial guess on the size of the proposal. 
 *    Is calibrated on first MCMC chain. Use 1 or pow(2, 1./ndim)
 * 
 * See \ref ultranest for documentation on the other parameters.
 */
ultranest_draw_state * ultranest_draw_init(LikelihoodFunc, 
	const unsigned int ndim, const unsigned int nsteps, 
	const double proposal_scale);

/**
 * Draw from Cube directly, reject points via RadFriends (optional).
 * 
 * \param current   point to update
 * \param members   live points for RadFriends
 * \param nmembers  number of live points
 * 
 * Set nmembers = 0 to draw from Cube without using RadFriends.
 */
unsigned int ultranest_draw_generate_direct(ultranest_draw_state * state, 
	point * current, const point ** members, const unsigned int nmembers);

/**
 * Draw a new point using the hybrid RadFriends/MCMC method.
 * 
 * \param current       point to update
 * \param live_points   live points for RadFriends
 * \param nlive_points  number of live points
 * 
 * Automatically switches over to MCMC when RadFriends becomes excessively 
 * inefficient.
 */
unsigned int ultranest_draw_next(ultranest_draw_state * state, point * current,
	const point ** live_points, const unsigned int nlive_points);

/**
 * Print some statistics on the performance of the MCMC proposal.
 */
void ultranest_draw_print_stats(const ultranest_draw_state * state);

#endif

