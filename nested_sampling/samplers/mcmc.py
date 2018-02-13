from __future__ import print_function
import scipy, scipy.stats
from numpy import exp, log, log10
import numpy

def is_inside_unit_filter(u):
	return numpy.all(u >= 0) and numpy.all(u <= 1)


class BaseProposal(object):
	"""
	Base class for proposal function.
	
	:param scale: Scale of proposal
	:param adapt: Adaptation rule to use for scale, when new_chain is called.
	
	If adapt is False, no adaptation is done. If adapt is 'Sivia', the rule
	of Sivia & Skilling (2006) is used. If adapt is something else,
	a crude thresholding adaptation is used to gain ~50% acceptance.
	"""
	def __init__(self, adapt = False, scale = 1.):
		self.accepts = []
		self.adapt = adapt
		self.scale = scale
	"""
	Proposal function (to be overwritten)
	"""
	def propose(self, u, ndim, live_pointsu=None, is_inside_filter=None):
		return u
	"""
	Reset accept counters and adapt proposal (if activated).
	"""
	def new_chain(self, live_pointsu=None, is_inside_filter=None):
		if self.adapt and len(self.accepts) > 0:
			# adjust future scale based on acceptance rate
			m = numpy.mean(self.accepts)
			assert 0 <= m <= 1, (m, self.accepts)
			if self.adapt == 'sivia':
				if m > 0.5: self.scale *= exp(1./numpy.sum(self.accepts))
				else:       self.scale /= exp(1./(len(self.accepts) - numpy.sum(self.accepts)))
			elif self.adapt == 'sivia-neg-binom':
				# negative binomial rate estimator
				m = (sum(self.accepts) - 1) / (len(self.accepts) - 1.)
				if m > 0.5: self.scale *= exp(1./numpy.sum(self.accepts))
				else:       self.scale /= exp(1./(len(self.accepts) - numpy.sum(self.accepts)))
			elif self.adapt == 'step':
				#print 'adaptation:', m
				if m <= 0.1:
					self.scale /= 1.1
				elif m <= 0.3:
					self.scale /= 1.01
				elif m >= 0.9:
					self.scale *= 1.1
				elif m >= 0.7:
					self.scale *= 1.01
			else:
				assert False, self.adapt
			assert numpy.all(numpy.isfinite(self.scale)), self.scale
		self.accepts = []
	
	"""
	Add a point to the record.
	:param accepted: True if accepted, False if rejected.
	"""
	def accept(self, accepted):
		self.accepts.append(accepted)
	
	"""
	Print some stats on the acceptance rate
	"""
	def stats(self):
		print('Proposal %s stats: %.2f%% accepts' % (repr(self), 
			numpy.mean(self.accepts) * 100.))

class MultiScaleProposal(BaseProposal):
	"""Proposal over multiple scales, inspired by DNest. 
	Uses the formula
	
	:math:`x + n * 10^{l - s * u}`
	
	where l is the location, s is the scale and u is a uniform variate,
	and n is a normal variate.
	
	@see MultiScaleProposal
	"""
	def __init__(self, loc = -4.5, scale=1.5, adapt=False):
		# 10**(1.5 - 6 * u) (inspired by DNest)
		# a + (b - a) * u
		# a = 1.5, b = -4.5
		# a should increase for larger scales, decrease for smaller
		
		self.loc = loc
		BaseProposal.__init__(self, adapt=adapt, scale=scale)
	def __repr__(self):
		return 'MultiScaleProposal(loc=%s, scale=%s, adapt=%s)' % (self.loc, self.scale, self.adapt)
	def propose(self, u, ndim, live_pointsu=None, is_inside_filter=None):
		p = u + numpy.random.normal() * 10**(self.scale + (self.loc - self.scale) * numpy.random.uniform())
		p[p > 1] = 1
		p[p < 0] = 0
		#p = p - numpy.floor(p)
		return p

class GaussProposal(BaseProposal):
	"""
	Symmetric gaussian proposal.

	@see BaseProposal
	"""
	def __init__(self, adapt = False, scale = 1.):
		BaseProposal.__init__(self, adapt=adapt, scale=scale)
	def propose(self, u, ndim, live_pointsu=None, is_inside_filter=None):
		p = u + numpy.random.normal(0, self.scale, size=ndim)
		# wrap around
		#p = p - numpy.floor(p)
		p[p > 1] = 1
		p[p < 0] = 0
		return p
	def __repr__(self):
		return 'GaussProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class MCMCConstrainer(object):
	"""
	Markov chain Monte Carlo proposals using the Metropolis update: 
	Do a number of steps, while adhering to boundary.
	"""
	def __init__(self, nsteps = 200, proposer = MultiScaleProposal(), nmaxsteps = 10000, nminaccepts=0):
		self.proposer = proposer
		self.sampler = None
		self.nsteps = nsteps
		self.nmaxsteps = nmaxsteps
		self.nminaccepts = nminaccepts
	
	def draw_constrained(self, Lmin, priortransform, loglikelihood, ndim, 
			startu, startx, startL, live_pointsu, **kwargs):
		ui = startu
		xi = startx
		Li = startL
		naccepts = 0
		assert Li >= Lmin
		while True:
			self.proposer.new_chain(ui, ndim, live_pointsu, is_inside_unit_filter)
			n = 0
			for i in range(self.nmaxsteps):
				u = self.proposer.propose(ui, ndim, live_pointsu, is_inside_unit_filter)
				x = priortransform(u)
				L = loglikelihood(x)
				n = n + 1
				# MH accept rule
				# accept = L > Li or numpy.random.uniform() < exp(L - Li)
				# Likelihood-difference independent, because we do
				# exploration of the prior (full diffusion).
				# but only accept in constrained region, because that
				# is what we are exploring now.
				accept = L >= Lmin
				if accept:
					ui, xi, Li = u, x, L
					naccepts += 1
			
				# tell proposer so it can scale
				self.proposer.accept(accept)
			
				if i + 1 >= self.nsteps and naccepts >= self.nminaccepts:
					if Li > Lmin:
						break
			if naccepts > 0:
				# if no accepts at all, restart from scratch.
				break
			else:
				print 'restarting failed chain...'
		if Li < Lmin:
			print()
			print('ERROR: MCMCConstrainer could not find a point matching constraint!')
			print('ERROR: Proposer stats:', self.proposer.stats())
			assert Li > Lmin, (Li, Lmin, self.nmaxsteps, numpy.mean(self.proposer.accepts), len(self.proposer.accepts))
		return ui, xi, Li, n

	def stats(self):
		return self.proposer.stats()

class SliceConstrainer(object):
	"""
	Markov chain Monte Carlo proposals using the Metropolis update: 
	Do a number of steps, while adhering to boundary.
	"""
	def __init__(self, nsteps = 200, proposer = MultiScaleProposal(), nmaxsteps = 10000, nminaccepts=0):
		self.proposer = proposer
		self.sampler = None
		self.nsteps = nsteps
		self.nmaxsteps = nmaxsteps
		self.nminaccepts = nminaccepts
	
	def draw_constrained(self, Lmin, priortransform, loglikelihood, ndim, 
			startu, startx, startL, live_pointsu, **kwargs):
		ui = startu
		xi = startx
		Li = startL
		naccepts = 0
		assert Li >= Lmin
		for i in range(self.nmaxsteps):
			self.proposer.new_chain(ui, ndim, live_pointsu, is_inside_unit_filter)
			n = 0
			while True:
				u = self.proposer.propose(ui, ndim, live_pointsu, is_inside_unit_filter)
				x = priortransform(u)
				L = loglikelihood(x)
				n = n + 1
				# MH accept rule
				# accept = L > Li or numpy.random.uniform() < exp(L - Li)
				# Likelihood-difference independent, because we do
				# exploration of the prior (full diffusion).
				# but only accept in constrained region, because that
				# is what we are exploring now.
				accept = L >= Lmin
				# tell proposer so it can scale
				self.proposer.accept(accept)
				if accept:
					ui, xi, Li = u, x, L
					naccepts += 1
					break
				assert n < self.nmaxsteps, n
		if Li < Lmin:
			print
			print 'ERROR: HARMConstrainer could not find a point matching constraint!'
			print 'ERROR: Proposer stats:',
			self.proposer.stats()
			assert Li > Lmin, (Li, Lmin, self.nmaxsteps, numpy.mean(self.proposer.accepts), len(self.proposer.accepts))
		return ui, xi, Li, n

	def stats(self):
		return self.proposer.stats()

