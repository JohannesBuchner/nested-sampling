import numpy
from numpy import exp, log, log10, pi

class RejectionConstrainer(object):
	"""
	Simplest draw function using rejection sampling.

	Guaranteed to yield uniform points *and* not to exclude any maxima. 
	Least efficient method.
	"""
	def __init__(self):
		self.sampler = None
	
	"""
	Return the found point with its likelihood, and the number of 
	likelihood calls performed.
	"""
	def draw_constrained(self, Lmin, priortransform, loglikelihood, draw_global_uniform, **kwargs):
		n = 0
		while True:
			u = draw_global_uniform()
			x = priortransform(u)
			L = loglikelihood(x)
			n = n + 1
			if Lmin is None or L > Lmin:
				return u, x, L, n

__all__ = [RejectionConstrainer]

