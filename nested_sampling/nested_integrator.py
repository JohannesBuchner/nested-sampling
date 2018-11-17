"""
Copyright: Johannes Buchner (C) 2013-2017

Modular, Pythonic Implementation of Nested Sampling
"""

from __future__ import print_function
import numpy
from numpy import exp, log, log10, pi
import progressbar
from .adaptive_progress import AdaptiveETA
from numpy import logaddexp


"""
Performs the Nested Sampling integration by calling the *sampler* multiple times
until the *tolerance* is reached, or the maximum number of likelihood evaluations
is exceeded.

:param sampler: Sampler
:param tolerance: uncertainty in log Z to compute to
:param max_samples: maximum number of likelihood evaluations (None for no limit)

@return dictionary containing the keys

  logZ, logZerr: log evidence and uncertainty, 
  samples: all obtained samples,
  weights: posterior samples: 
  	list of prior coordinates, transformed coordinates, likelihood value 
  	and weight
  information: information H
  niterations: number of nested sampling iterations
"""
def nested_integrator(sampler, terminationcriterion, check_every=10,
	max_samples=None):
	logVolremaining = 0
	logwidth = log(1 - exp(-1. / sampler.nlive_points))
	weights = [] #[-1e300, 1]]
	
	convergence_tests = []
	widgets = ['|...|',
		progressbar.Bar(), progressbar.Percentage(), progressbar.AdaptiveETA()]
	pbar = progressbar.ProgressBar(widgets = widgets)
	
	i = 0
	ui, xi, Li = next(sampler)
	wi = logwidth + Li
	logZ = wi
	H = Li - logZ
	pbar.currval = i
	max_value = sampler.nlive_points
	pbar.start()
	while True:
		i = i + 1
		logwidth = log(1 - exp(-1. / sampler.nlive_points)) + logVolremaining
		logVolremaining -= 1. / sampler.nlive_points
		
		weights.append([ui, xi, Li, logwidth])
		
		logZerr = (H / sampler.nlive_points)**0.5
		
		pbar.update(i)
		
		# expected number of iterations:
		i_final = -sampler.nlive_points * (-sampler.Lmax + log(exp(max(0.5 - logZerr, logZerr / 100.) + logZ) - exp(logZ) + 1e-300))
		max_value = min(max(i+1, i_final), i+100000)
		if hasattr(pbar, 'max_value'): pbar.max_value = max_value
		if hasattr(pbar, 'maxval'):    pbar.maxval = max_value
		
		if i == 1 or (i > sampler.nlive_points and i % check_every == 1):
			terminationcriterion.update(sampler, logwidth, logVolremaining, logZ, H, sampler.Lmax)
			
			total_error = terminationcriterion.totalZerr
			if max_samples is not None and int(max_samples) < int(sampler.ndraws):
				pbar.finish()
				print('maximum number of samples reached')
				break
			if terminationcriterion.converged:
				pbar.finish()
				print('tolerance on error reached: total=%.4f stat=%.4f remainder=%.4f' % (terminationcriterion.totalZerr, terminationcriterion.normalZerr, terminationcriterion.remainderZerr))
				break
			# we want to make maxContribution as small as possible
			#   but if it becomes 10% of logZerr, that is enough
			if terminationcriterion.remainderZerr < logZerr / 10.:
				pbar.finish()
				print('tolerance will not improve: remainder error (%.3f) is much smaller than systematic errors (%.3f)' % (terminationcriterion.remainderZerr, logZerr))
				break
		
		widgets[0] = '|%d/%d samples+%d/%d|lnZ = %.2f +- %.3f + %.3f|L=%.2f @ %s' % (
			i + 1, max_value, sampler.nlive_points, sampler.ndraws, terminationcriterion.totalZ, logZerr, terminationcriterion.remainderZerr, Li,
			numpy.array_str(xi, max_line_width=1000, precision=4))
		ui, xi, Li = next(sampler)
		wi = logwidth + Li
		logZnew = logaddexp(logZ, wi)
		H = exp(wi - logZnew) * Li + exp(logZ - logZnew) * (H + logZ) - logZnew
		logZ = logZnew
	
	# not needed for integral, but for posterior samples, otherwise there
	# is a hole in the most likely parameter ranges.
	weights += [[ui, xi, Li, logwidth] for ui, xi, Li in sampler.remainder()]
	totalZerr = terminationcriterion.totalZerr
	return dict(logZ=terminationcriterion.totalZ, logZerr=totalZerr, 
		samples=sampler.samples, weights=weights, information=H,
		niterations=i) #, convergence_tests=convergence_tests)

__all__ = [nested_integrator]

