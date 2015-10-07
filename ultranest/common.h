#ifndef ULTRANEST_COMMON
#define ULTRANEST_COMMON

/** MultiNest compatible Likelihood function **/
#define LikelihoodFunc void (*Like) (double * Cube, int * ndim, int * nparams, double * lnew, void * userparams)

/** data structure for points **/
typedef struct {
	/// point coordinates on unit cube
	double * coords;
	/// "physical" point coordinates, i.e. transformed
	double * phys_coords;
	/// likelihood value
	double L;
} point;

/**
 * Point with weight.
 */
typedef struct {
	point * p;
	double weight;
} weighted_point;

/** helper functions: **/

/**
 * Allocate a new point
 */
point * create_point(const unsigned int ndim);

/**
 * De-allocate a point
 */
void free_point(point * p);

/** adding two log numbers **/
double logaddexp(const double a, const double b);
/** subtracting two log numbers **/
double logsubexp(const double a, const double b);

/** euclidean distance between two points **/
double compute_distance(const unsigned int ndim, const point * member, const point * other);
/** computing maxdistance from scratch, for RadFriends **/
double nearest_rdistance_guess(const unsigned int ndim, const point ** live_points, const unsigned int nlive_points);

#endif
