import numpy
from numpy import exp, log, log10, pi
from nestle import bounding_ellipsoid, bounding_ellipsoids, sample_ellipsoids

class EllipsoidConstrainer(object):
	"""
	Simplest draw function using rejection sampling.

	Guaranteed to yield uniform points *and* not to exclude any maxima. 
	Least efficient method.
	"""
	def __init__(self, enlarge=1.2, update_interval=None):
		self.sampler = None
		self.enlarge = enlarge
		self.iter = 0
		self.ells = None
		self.update_interval = update_interval
	
	def update(self, points):
		pointvol = exp(-self.iter / len(points)) / len(points)
		self.ell = bounding_ellipsoid(numpy.asarray(points), pointvol=pointvol,
			minvol=True)
		self.ell.scale_to_vol(self.ell.vol * self.enlarge)
	
	"""
	Return the found point with its likelihood, and the number of 
	likelihood calls performed.
	"""
	def draw_constrained(self, Lmin, priortransform, loglikelihood, draw_global_uniform, live_pointsu, ndim, **kwargs):
		update_interval = max(1, round(0.2 * len(live_pointsu)))
		if self.iter % update_interval == 0:
			self.update(live_pointsu)
		self.iter += 1
		n = 0
		while True:
			u = self.ell.sample(rstate=numpy.random)
			if not (numpy.all(u > 0.) and numpy.all(u < 1.)):
				continue
			x = priortransform(u)
			L = loglikelihood(x)
			n = n + 1
			if Lmin is None or L > Lmin:
				return u, x, L, n


class MultiEllipsoidConstrainer(object):
	"""
	Simplest draw function using rejection sampling.

	Guaranteed to yield uniform points *and* not to exclude any maxima. 
	Least efficient method.
	"""
	def __init__(self, enlarge=1.2, update_interval=50):
		self.sampler = None
		self.enlarge = enlarge
		self.iter = 0
		self.update_interval = update_interval
	
	def update(self, points):
		pointvol = exp(-self.iter / len(points)) / len(points)
		self.ells = bounding_ellipsoids(numpy.asarray(points), pointvol=pointvol)
		for ell in self.ells:
			ell.scale_to_vol(ell.vol * self.enlarge)
	
	"""
	Return the found point with its likelihood, and the number of 
	likelihood calls performed.
	"""
	def draw_constrained(self, Lmin, priortransform, loglikelihood, draw_global_uniform, live_pointsu, ndim, **kwargs):
		update_interval = max(1, round(0.2 * len(live_pointsu)))
		if self.iter % update_interval == 0:
			self.update(live_pointsu)
		self.iter += 1
		
		n = 0
		while True:
			u = sample_ellipsoids(self.ells, rstate=numpy.random)
			if not (numpy.all(u > 0.) and numpy.all(u < 1.)):
				continue
			x = priortransform(u)
			L = loglikelihood(x)
			n = n + 1
			if Lmin is None or L > Lmin:
				return u, x, L, n

__all__ = [EllipsoidConstrainer, MultiEllipsoidConstrainer]

