#ifndef ULTRANEST_SAMPLER
#define ULTRANEST_SAMPLER

#include "common.h"
#include "draw.h"

/**
 * \file
 * \brief The Sampler keeps a list of live points and replaces the worst live 
 *       points in each iteration.
 */

/**
 * Information about which likelihood constrain a point was drawn from,
 * to continue previous runs.
 */
typedef struct {
	/// Coordinates
	point * p;
	/// Likelihood constraint this point was drawn from
	double Lmin;
} draw_point;

/**
 * Private variables of the sampler.
 */
typedef struct {
	/// Prefix of output files
	const char * root;
	/// counter for number of draws
	unsigned int ndraws;
	/// number of live points
	unsigned int nlive_points;
	/// dimensionality
	unsigned int ndim;
	/// current remainder integral
	double remainderZ;
	/// uncertainty on \ref remainderZ
	double remainderZerr;
	/// highest likelihood encountered
	double Lmax;
	/// Likelihood function
	LikelihoodFunc;
	/// Storage for live points
	point ** live_points;
	/// State of drawing method.
	ultranest_draw_state * draw_state;
	/// Storage of last run
	draw_point * draws_prev;
	/// Size of storage from last run
	unsigned int draws_prev_n;
	/// Output of draws from current run
	FILE * draws_file;
} ultranest_state;

/**
 * Create sampler.
 * 
 * \param draw_state state of the drawing method. Created with \ref ultranest_draw_init
 * See \ref ultranest for documentation on the other parameters.
 * 
 * Uses \ref ultranest_draw_generate_direct to initialise the first live points.
 */
ultranest_state * ultranest_sampler_init(LikelihoodFunc, 
	const char * root, int ndim, unsigned int nlive_points, 
	ultranest_draw_state * draw_state);

/**
 * Next sampler iteration. Will remove least likely point and find a higher
 * replacement, using the drawing method.
 * 
 * Uses \ref ultranest_draw_next
 */
point * ultranest_sampler_next(ultranest_state * state);

/**
 * Integrate the current live points (remainder on the integral).
 * Step integration is used.
 * 
 * \param state             Sampler
 * \param logwidth          current point weight
 * \param logVolremaining   current remaining volume
 * \param logZ              current evidence
 * \param points            if given, stores the live points with weights into 
 *                          this array.
 *
 * @return number of live points
 * 
 */
int ultranest_sampler_integrate_remainder(ultranest_state * state, 
	const double logwidth, const double logVolremaining, const double logZ, 
	weighted_point * points);

#endif

