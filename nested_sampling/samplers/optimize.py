from __future__ import print_function
import scipy, scipy.optimize
import numpy
from numpy import exp, log, log10
import numpy

class OptimizeConstrainer(object):
	"""
	Very simple improvement to Nested Sampling:
	to avoid premature termination, we find the maximum with a optimization
	algorithm to be better able to determine how much evidence could
	still be missing.
	"""
	def __init__(self, optimizer = scipy.optimize.fmin):
		self.optimizer = optimizer
		self.sampler = None
	
	def draw_constrained(self, Lmin, priortransform, loglikelihood, previous, ndim, **kwargs):
		previousL = numpy.array([L for _, _, L in previous])
		previousu = numpy.array([u for u, _, _ in previous])
		i = numpy.argmax(previousL)
		ustart = previousu[i]
		Lstart = previousL[i]
		
		if self.sampler.Lmax < Lstart:
			def minfunc(ui):
				return -loglikelihood(priortransform(ui))
			ubest = self.optimizer(minfunc, ustart)
			Lbest = loglikelihood(priortransform(ubest))
			print('old best:', ustart, Lstart)
			print('new best:', ubest, Lbest)
			if self.sampler:
				self.sampler.Lmax = max(self.sampler.Lmax, Lbest)
		n = 0
		while True:
			u = numpy.random.uniform(0,1, size=ndim)
			x = priortransform(u)
			L = loglikelihood(u)
			n = n + 1
			if L > Lmin:
				return u, x, L, n
__all__ = [OptimizeConstrainer]

