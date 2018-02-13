from __future__ import print_function
"""
Copyright: Johannes Buchner (C) 2013

Modular, Pythonic Implementation of Nested Sampling
"""

import numpy
from numpy import exp, log, log10, pi
from operators import itemgetter

class AdaptiveNestedSampler(object):
	"""
	Samples points, always replacing the worst live point, forever
	
	The number of live points is variable, but at most nlive_points. 
	When the loglikelihoods of the last two points are the same,
	the number of live points is reduced, down to nlive_points_min.
	When they are different, live points are added.
	
	expected_steps dictates the required steepness for expansion, i.e.
	we expect (Lmax - L0) / expected_steps = L0 - L1 for good progress
	and completion within expected_steps iterations.
	"""
	def __init__(self, priortransform, loglikelihood, draw_constrained, 
		ndim=1, nlive_points = 1000, nlive_points_min = 50, 
		expected_steps = 100000):
		self.nlive_points_max = nlive_points
		self.nlive_points_min = nlive_points
		self.priortransform = priortransform
		self.loglikelihood = loglikelihood
		self.draw_constrained = draw_constrained
		self.expected_steps = expected_steps
		self.samples = []
		self.ndim = ndim
		# draw N starting points from prior
		live_points = []
		for i in range(nlive_points):
			u = numpy.random.uniform(0, 1, size=ndim)
			assert len(u) == ndim, (u, ndim)
			x = priortransform(u)
			assert len(x) == ndim, (x, ndim)
			L = loglikelihood(x)
			live_points.append((L, u, x))
			self.samples.append([u, x, L])
		self.live_points = live_points
		self.nlive_points = len(self.live_points)
		self.Lmax = max([L for L, u, x in live_points])
		self.ndraws = nlive_points
	
	def add_point(self, Lmin):
		# choose start point at random 
		k = numpy.random.randint(0, self.nlive_points - 1)
		
		# find replacement
		uj, xj, Lj, n = self.draw_constrained(
			Lmin=Lmin, 
			priortransform=self.priortransform, 
			loglikelihood=self.loglikelihood, 
			previous=self.samples,
			ndim=self.ndim,
			startu = self.live_points[k][1], 
			startx = self.live_points[k][2], 
			startL = self.live_points[k][0],
			starti = k)
		
		self.live_points.append((Lj, uj, xj))
		self.Lmax = max(Lj, self.Lmax)
		self.samples.append([uj, xj, Lj])
		self.ndraws += int(n)
	
	def __next__(self):
		# nlive_points has to tell how many points were used when
		# the last sample was returned
		self.nlive_points = len(self.live_points)
				
		# select worst point
		self.live_points.sort(key=itemgetter(0))
		Li, ui, xi = self.live_points.pop()
		
		Lsecond = self.live_points[-1][0]
		
		#if (Lsecond - Li) < (self.Lmax - Li) * self.expected_steps:
		# current iteration of nested integrator
		# it = self.samples - self.nlive_points
		r = (1 - exp(Lsecond - Li)) / (1 - exp(self.Lmax - Li)) / (1 - exp(-1./self.nlive_points))
		if r < 1e-3:
			# there is a very flat curve
			#   do not add a point (contract)
			print('Adaption: contracting')
			pass
		elif r > 1:
			# currently have a steep curve and maximum not exceeded
			#   then fill up (expand)
			# this step can be parallelized
			print('Adaption: expanding')
			while len(self.live_points) < self.nlive_points_max:
				self.add_point(Lmin=Li)
		else:
			# reasonably flat, keep steady state
			if len(self.live_points) < self.nlive_points_max:
				self.add_point(Lmin=Li)
		
		# make sure we have at least the minimum number of points
		while len(self.live_points) < self.nlive_points_min:
			self.add_point(Lmin=Li)
		
		return ui, xi, Li
	def remainder(self):
		for L, u, x in self.live_points:
			yield u, x, L
	def __next__(self):
		return self.__next__()
	def __iter__(self):
		while True: yield self.__next__()
		
__all__ = [AdaptiveNestedSampler]

