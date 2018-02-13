import numpy
import scipy.spatial, scipy.cluster
import matplotlib.pyplot as plt
from .friends import FriendsConstrainer
from .mcmc import BaseProposal
import numpy
from numpy import exp, log, log10, pi, cos, sin
from nestle import bounding_ellipsoid, bounding_ellipsoids, sample_ellipsoids

class FilteredGaussProposal(BaseProposal):
	"""
	Symmetric gaussian proposal.

	@see BaseProposal
	"""
	def __init__(self, adapt = True, scale = 1.):
		BaseProposal.__init__(self, adapt=adapt, scale=float(scale))
	
	def new_chain(self, u, ndim, points, is_inside_filter):
		BaseProposal.new_chain(self)
	
	def new_direction(self, u, ndim, points, is_inside_filter):
		pass
	
	def propose(self, u, ndim, points, is_inside_filter):
		while True:
			p = u + numpy.random.normal(0, self.scale, size=ndim)
			if not is_inside_filter(p):
				# narrow down
				self.accept(False)
			return p
	
	def __repr__(self):
		return 'FilteredGaussProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class FilteredGibbsProposal(FilteredGaussProposal):
	"""
	Symmetric gaussian, component-wise proposal.

	@see FilteredGaussProposal
	"""
	def propose(self, u, ndim, points, is_inside_filter):
		while True:
			i = numpy.random.randint(ndim)
			p = u.copy()
			p[i] += numpy.random.normal(0, self.scale)
			if not is_inside_filter(p):
				# narrow down
				self.accept(False)
			return p

class FilteredSVarGaussProposal(BaseProposal):
	"""
	Gaussian proposal, scaled by std of live points in each dimension.

	@see BaseProposal
	"""
	def __init__(self, adapt = True, scale = 1.):
		BaseProposal.__init__(self, adapt=adapt, scale=float(scale))
	
	def new_chain(self, u, ndim, points, is_inside_filter):
		BaseProposal.new_chain(self)
		self.axes_scale = numpy.std(points, axis=0)
		#print 'scale:', self.axes_scale, self.scale
		assert self.axes_scale.shape == (ndim,), self.axes_scale.shape
	
	def new_direction(self, u, ndim, points, is_inside_filter):
		pass
	
	def propose(self, u, ndim, points, is_inside_filter):
		while True:
			p = numpy.random.normal(u, self.scale * self.axes_scale)
			if is_inside_filter(p):
				return p
			else:
				# narrow down
				self.accept(False)
	
	def __repr__(self):
		return 'FilteredSVarGaussProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class FilteredSVarGibbsProposal(FilteredSVarGaussProposal):
	"""
	Component-wise Gaussian proposal, scaled by std of live points in each dimension.

	@see FilteredSVarGaussProposal
	"""
	def propose(self, u, ndim, points, is_inside_filter):
		while True:
			i = numpy.random.randint(ndim)
			p = u.copy()
			p[i] += numpy.random.normal(0, self.scale * self.axes_scale[i])
			if is_inside_filter(p):
				return p
			else:
				# narrow down
				self.accept(False)

class FilteredMahalanobisGaussProposal(BaseProposal):
	"""
	Gaussian proposal with Mahalanobis metric from live points.

	@see BaseProposal
	"""
	def __init__(self, adapt = True, scale = 1.):
		BaseProposal.__init__(self, adapt=adapt, scale=float(scale))
	
	def new_chain(self, u, ndim, points, is_inside_filter):
		BaseProposal.new_chain(self)
		self.metric = numpy.cov(numpy.transpose(points))
		assert self.metric.shape == (ndim,ndim), self.metric.shape
	
	def new_direction(self, u, ndim, points, is_inside_filter):
		pass
	
	def propose(self, u, ndim, points, is_inside_filter):
		while True:
			p = numpy.random.multivariate_normal(u, self.scale * self.metric)
			if is_inside_filter(p):
				return p
			else:
				# narrow down
				self.accept(False)
	
	def __repr__(self):
		return 'FilteredMahalanobisGaussProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class FilteredEigVecGibbsProposal(FilteredMahalanobisGaussProposal):
	"""
	Eigenvector-wise Gaussian proposal

	@see FilteredMahalanobisGaussProposal
	"""
	def new_chain(self, u, ndim, points, is_inside_filter):
		BaseProposal.new_chain(self)
		self.metric = numpy.cov(numpy.transpose(points))
		assert self.metric.shape == (ndim,ndim), self.metric.shape
		w, v = numpy.linalg.eig(self.metric)
		self.eigvec = w * v
		assert self.eigvec.shape == (ndim, ndim), self.eigvec.shape
	
	def propose(self, u, ndim, points, is_inside_filter):
		while True:
			i = numpy.random.randint(ndim)
			p = u.copy()
			p += numpy.random.normal(0, self.scale) * self.eigvec[i]
			if is_inside_filter(p):
				return p
			else:
				# narrow down
				self.accept(False)

class FilteredUnitHARMProposal(BaseProposal):
	"""
	Unit HARM proposal.

	@see BaseProposal
	"""
	def __init__(self, adapt = False, scale = 1.):
		BaseProposal.__init__(self, adapt=False, scale=float(scale))
	
	def generate_direction(self, u, ndim, points):
		# generate unit direction
		x = numpy.random.normal(size=ndim)
		d = x / (x**2).sum()**0.5
		return d
	def new_chain(self, u, ndim, points, is_inside_filter):
		BaseProposal.new_chain(self)
		self.new_direction(u, ndim, points, is_inside_filter)
	def new_direction(self, u, ndim, points, is_inside_filter):
		d = self.generate_direction(u, ndim, points)
		print('initial scale:', self.scale)
		# find end points
		forward_scale = self.scale
		# find a scale that is too large
		while True:
			assert forward_scale > 0
			p_for = u + d * forward_scale
			if is_inside_filter(p_for):
				# we are proposing too small. We should be outside
				forward_scale *= 2
				print('too small, stepping further', forward_scale)
			else:
				break
		
		backward_scale = self.scale
		# find a scale that is too large
		while True:
			assert backward_scale > 0
			p_rev = u - d * backward_scale
			if is_inside_filter(p_rev):
				# we are proposing too small. We should be outside
				print('too small, stepping back', backward_scale)
				backward_scale *= 2
			else:
				break
		# remember scale for next time:
		self.backward_scale = -backward_scale
		self.forward_scale = forward_scale
		self.direction = d
	
	def propose(self, u, ndim, points, is_inside_filter):
		# generate a random point between the two points.
		while True:
			print('slice range:', (self.backward_scale, self.forward_scale))
			x = numpy.random.uniform(self.backward_scale, self.forward_scale)
			p = u + self.direction * x
			assert self.forward_scale - self.backward_scale > 1e-100
			if x < 0:
				self.backward_scale = x
			else:
				self.forward_scale = x
			if is_inside_filter(p):
				if self.adapt:
					self.scale = self.forward_scale - self.backward_scale
					print('adapting scale to', self.scale)
				return p
	
	def accept(self, accepted):
		# scale should not be modified
		pass
	
	def __repr__(self):
		return 'FilteredUnitHARMProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class FilteredUnitIterateSliceProposal(FilteredUnitHARMProposal):
	"""
	Unit Slice sampling proposal, iterative component-wise.

	@see BaseProposal
	"""
	def __init__(self, adapt = False, scale = 1.):
		BaseProposal.__init__(self, adapt=False, scale=float(scale))
		self.curindex = 0
	
	def generate_direction(self, u, ndim, points):
		# choose next base vector
		d = numpy.zeros(ndim)
		self.curindex = (self.curindex + 1) % ndim
		d[self.curindex] = 1
		return d
	def __repr__(self):
		return 'FilteredUnitIterateSliceProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class FilteredUnitRandomSliceProposal(FilteredUnitHARMProposal):
	"""
	Unit Slice sampling proposal, random component-wise.

	@see BaseProposal
	"""
	def generate_direction(self, u, ndim, points):
		# choose a random base vector
		d = numpy.zeros(ndim)
		i = numpy.random.randint(ndim)
		d[i] = 1
		return d
	def __repr__(self):
		return 'FilteredUnitRandomSliceProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class FilteredMahalanobisHARMProposal(FilteredUnitHARMProposal):
	"""
	Mahalanobis HARM proposal.

	@see BaseProposal
	"""

	def generate_direction(self, u, ndim, points):
		# generate direction from mahalanobis metric
		metric = numpy.cov(numpy.transpose(points))
		assert metric.shape == (ndim,ndim), metric.shape
		x = numpy.random.multivariate_normal(numpy.zeros(ndim), metric)
		d = x / (x**2).sum()**0.5
		return d
	def __repr__(self):
		return 'FilteredMahalanobisHARMProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)


class FilteredPointHARMProposal(FilteredUnitHARMProposal):
	"""
	HARM proposal using live points

	@see BaseProposal
	"""
	def generate_direction(self, u, ndim, points):
		# draw direction from points
		while True:
			i = numpy.random.randint(len(points))
			if numpy.all(points[i,:] == u): 
				continue
			x = points[i,:] - u
			d = x / (x**2).sum()**0.5
			return d
	def __repr__(self):
		return 'FilteredPointHARMProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)


class FilteredDeltaPointHARMProposal(FilteredUnitHARMProposal):
	"""
	HARM proposal using live points

	@see BaseProposal
	"""
	def generate_direction(self, u, ndim, points):
		# draw direction from points
		while True:
			i = numpy.random.randint(len(points))
			if numpy.all(points[i,:] == u): 
				continue
			j = numpy.random.randint(len(points) - 1)
			if j >= i:
				j += 1
			x = points[i,:] - points[j,:]
			d = x / (x**2).sum()**0.5
			return d
	def __repr__(self):
		return 'FilteredDeltaPointHARMProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)


class FilteredEllipticalSliceProposal(BaseProposal):
	"""
	Elliptical Slice proposal using live points

	@see BaseProposal
	"""
	def __init__(self):
		BaseProposal.__init__(self, adapt=False, scale=1.)
	
	def new_direction(self, u, ndim, points, is_inside_filter):
		# 1. choose ellipse
		self.v = points[numpy.random.randint(len(points)),:]
		self.center = numpy.mean(points, axis=0)
		# threshold is fixed
		# define bracket
		self.theta = numpy.random.uniform(0, 2 * numpy.pi)
		self.theta_min = self.theta - 2*pi
		self.theta_max = self.theta
	
	def new_chain(self, u, ndim, points, is_inside_filter):
		BaseProposal.new_chain(self)
		self.new_direction(u, ndim, points, is_inside_filter)
	
	def propose(self, u, ndim, points, is_inside_filter):
		while True:
			p = (u - self.center) * cos(self.theta) + (self.v - self.center) * sin(self.theta) + self.center
			# prepare for a rejection:
			# shrink bracket
			
			# if we accept, we never re-enter this function
			# and the shrinkage does not matter
			
			if self.theta < 0:
				self.theta_min = self.theta
			else:
				self.theta_max = self.theta
			self.theta = numpy.random.uniform(self.theta_min, self.theta_max)
			
			if is_inside_filter(p):
				return p
	
	def __repr__(self):
		return 'FilteredEllipsoidalSliceProposal(scale=%s, adapt=%s)' % (self.scale, self.adapt)

class FilteredMCMCConstrainer(object):
	"""
	Markov chain Monte Carlo proposals using the Metropolis update: 
	Do a number of steps, while adhering to boundary.
	"""
	def __init__(self, proposer, nsteps = 200, nmaxsteps = 10000, nminaccepts=0):
		self.proposer = proposer
		self.sampler = None
		self.nsteps = nsteps
		self.nmaxsteps = nmaxsteps
		self.nminaccepts = nminaccepts
	
	def draw_constrained(self, Lmin, priortransform, loglikelihood, live_pointsu, ndim, live_points_and_phantoms_u, is_inside_filter, startu, startx, startL):
		ui, xi, Li = startu, startx, startL
		while True:
			self.proposer.new_chain(ui, ndim, live_points_and_phantoms_u, is_inside_filter)
			n = 0
			naccepts = 0
			for i in range(self.nmaxsteps):
				u = self.proposer.propose(ui, ndim, live_points_and_phantoms_u, is_inside_filter)
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
					self.proposer.new_direction(ui, ndim, live_points_and_phantoms_u, is_inside_filter)
					naccepts += 1
			
				# tell proposer so it can scale
				self.proposer.accept(accept)
			
				if i + 1 >= self.nsteps and naccepts >= self.nminaccepts:
					if Li > Lmin:
						self.start_finish_check(ui, startu, live_pointsu)
						return ui, xi, Li, n
			if naccepts > 0:
				break
			else:
				print('restarting failed chain...')
		#print 'accepted %d' % naccepts
		#self.proposer.stats()
		if Li < Lmin:
			print()
			print('ERROR: HybridMCMCConstrainer could not find a point matching constraint!')
			print('ERROR: Proposer stats:')
			self.proposer.stats()
			assert Li >= Lmin, (Li, Lmin, self.nmaxsteps, numpy.mean(self.proposer.accepts), len(self.proposer.accepts))
		self.start_finish_check(ui, startu, live_pointsu)
		return ui, xi, Li, n
	
	def start_finish_check(self, ui, startu, live_pointsu):
		pass

	def stats(self):
		return self.proposer.stats()

class FilteredSliceConstrainer(object):
	"""
	Markov chain Monte Carlo proposals using the Metropolis update: 
	Do a number of steps, while adhering to boundary.
	"""
	def __init__(self, proposer, nsteps = 200):
		self.proposer = proposer
		self.sampler = None
		self.nsteps = nsteps
	
	def draw_constrained(self, Lmin, priortransform, loglikelihood, live_pointsu, ndim, live_points_and_phantoms_u, is_inside_filter, startu, startx, startL):
		ui, xi, Li = startu, startx, startL
		assert Li >= Lmin, (Li, Lmin)
		print('new chain...', Lmin, Li, startu)
		self.proposer.new_chain(ui, ndim, live_points_and_phantoms_u, is_inside_filter)
		for i in range(self.nsteps):
			n = 0
			naccepts = 0
			while True:
				print('proposing ...', ui)
				u = self.proposer.propose(ui, ndim, live_points_and_phantoms_u, is_inside_filter)
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
					print('accepting; new direction.', Lmin, u, L)
					self.proposer.new_direction(ui, ndim, live_points_and_phantoms_u, is_inside_filter)
					break
			
			
		#print 'accepted %d' % naccepts
		#self.proposer.stats()
		if Li < Lmin:
			print()
			print('ERROR: FilteredSliceConstrainer could not find a point matching constraint!')
			print('ERROR: Proposer stats:')
			self.proposer.stats()
			assert Li >= Lmin, (Li, Lmin, self.nsteps, numpy.mean(self.proposer.accepts), len(self.proposer.accepts))
		self.start_finish_check(ui, startu, live_pointsu)
		return ui, xi, Li, n
	
	def start_finish_check(self, ui, startu, live_pointsu):
		pass

	def stats(self):
		return self.proposer.stats()


class FilteredVarlengthMCMCConstrainer(FilteredMCMCConstrainer):
	"""
	Markov chain Monte Carlo proposals using the Metropolis update: 
	Do a number of steps, while adhering to boundary.
	"""
	def __init__(self, proposer, nsteps_initial = 2, nmaxsteps = 10000):
		FilteredMCMCConstrainer.__init__(self, proposer=proposer, nminaccepts=nsteps_initial, nsteps=nsteps_initial, nmaxsteps = nmaxsteps)
		self.last_was_start_memory = []

	def start_finish_check(self, ui, startu, live_pointsu):
		# is final point closest live point
		i = numpy.argmin(((live_pointsu - ui)**2).sum(axis=1))
		j = numpy.argmin(((live_pointsu - startu)**2).sum(axis=1))
		self.last_was_start_memory.append(i == j)
		
		# this should happen exactly once every len(live_pointsu)
		#lam = 1. / len(live_pointsu)
		lam = 1. / 5
		memory_part = self.last_was_start_memory[-4*len(live_pointsu):]
		n = len(memory_part)
		k = sum(memory_part)
		prob = scipy.stats.binom.pmf(k, n, lam)
		if k > lam * n and prob < 2.87e-7: # 5 sigma
			self.nsteps = self.nsteps * 2
			self.nminaccepts = self.nminaccepts * 2
			print('Saw %d/%d start~end cases, expected %.1f, probability is %.2e --> extending MCMC chains to %d steps' % (k, n, lam*n, prob, self.nsteps))
			self.last_was_start_memory = []
	
	def stats(self):
		print('FilteredVarlengthMCMCConstrainer: final nsteps was %d' % self.nsteps)
		return self.proposer.stats()

class HybridFriendsConstrainer(object):
	"""
	Do MCMC within RadFriends constraints
	"""
	def __init__(self, friends, mcmc_proposer, verbose=False, switchover_efficiency=0):
		self.friends = friends
		self.iter = 0
		self.mcmc_proposer = mcmc_proposer
		self.verbose = verbose
		self.switchover_efficiency = switchover_efficiency
		self.use_direct_draw = switchover_efficiency > 0

	def get_Lmax(self):
		return self.friends.get_Lmax()
	
	def draw_constrained(self, Lmin, priortransform, loglikelihood, live_pointsu, live_pointsx, live_pointsL, ndim, startu, startx, startL, **kwargs):
		if self.use_direct_draw:
			u, x, L, ntoaccept = self.friends.draw_constrained(Lmin=Lmin, priortransform=priortransform, loglikelihood=loglikelihood, live_pointsu=live_pointsu, ndim=ndim, startu=startu, startx=startx, startL=startL, **kwargs)
			if 1. / ntoaccept < self.switchover_efficiency:
				print('low efficiency triggered switching over to MCMC')
				self.use_direct_draw = False
			return u, x, L, ntoaccept
		
		ntoaccept, ntotalsum, rebuild = self.friends._draw_constrained_prepare(Lmin=Lmin, priortransform=priortransform, loglikelihood=loglikelihood, live_pointsu=live_pointsu, live_pointsx=live_pointsx, live_pointsL=live_pointsL, ndim=ndim, **kwargs)
		
		while True:
			# choose a new start point
			# can not be a phantom point, because those are below the threshold
			u, x, L, n = self.mcmc_proposer.draw_constrained(Lmin, priortransform, loglikelihood, live_pointsu, ndim, self.friends.region['members'], self.friends.is_inside, startu, startx, startL)
			ntoaccept += n
			
			if self.friends.is_inside(u) and L > Lmin:
				assert (u >= 0).all() and (u <= 1).all(), u
				# yay, we win
				if ntotalsum > 10000: 
					if self.verbose: 
						print('sampled %d points, evaluated %d ' % (ntotalsum, ntoaccept))
						#self.debugplot(u)
				#print 'returning:', u, x, L, ntoaccept
				return u, x, L, ntoaccept
			print('unsuccessful search; trying different starting point')
			# unsuccessful search so far. Try a different start point.
			starti = numpy.random.randint(len(live_pointsu))
			startu = live_pointsu[starti]
			startx = live_pointsx[starti]
			startL = live_pointsL[starti]
			
			# if running very inefficient, optimize clustering 
			#     if we haven't done so at the start
			if not rebuild and ntoaccept > 1000:
				rebuild = True
				self.friends.rebuild(numpy.asarray(live_pointsu), ndim, keepRadius=False)

	def stats(self):
		print('HybridFriendsConstrainer: %s' % ('used direct draws throughout (no MCMC)' if self.use_direct_draw else 'switched over to MCMC'))
		return self.mcmc_proposer.stats()

class HybridMLFriendsConstrainer(object):
	"""
	Do MCMC within RadFriends constraints, with metric-learning.
	
	This is EAGLENEST-safe.

	friends: MLFriendsConstrainer

	mcmc_proposer: FilteredMCMCConstrainer(nsteps=5, nminaccepts=5) with e.g. FilteredUnitHARMProposal()
	
	switchover_efficiency: if 0, always use MCMC. 
	    Otherwise, use direct draws from multiellipsoids. If this becomes
	    inefficient (rejection rate below switchover_efficiency),
	    use MCMC for the remainder of the run.
	
	"""
	def __init__(self, friends, mcmc_proposer, verbose=False, 
		switchover_efficiency=0, unfiltered=False):
		self.friends = friends
		self.iter = 0
		self.mcmc_proposer = mcmc_proposer
		self.verbose = verbose
		self.switchover_efficiency = switchover_efficiency
		self.use_direct_draw = switchover_efficiency > 0
		self.unfiltered = unfiltered

	def get_Lmax(self):
		return self.friends.get_Lmax()
	
	def is_inside(self, u):
		if self.unfiltered:
			return ((u >= 0).all() and (u <= 1).all())
		else:
			return self.friends.is_inside(u)
	def draw_constrained(self, Lmin, priortransform, loglikelihood, live_pointsu, live_pointsx, live_pointsL, ndim, startu, startx, startL, **kwargs):
		if self.use_direct_draw and self.friends.direct_draws_efficient:
			u, x, L, ntoaccept = self.friends.draw_constrained(Lmin=Lmin, priortransform=priortransform, loglikelihood=loglikelihood, live_pointsu=live_pointsu, ndim=ndim, startu=startu, startx=startx, startL=startL, **kwargs)
			if 1. / ntoaccept < self.switchover_efficiency:
				print('low efficiency triggered switching over to MCMC')
				self.use_direct_draw = False
			return u, x, L, ntoaccept
		
		ntoaccept, ntotalsum, rebuild = self.friends._draw_constrained_prepare(Lmin=Lmin, priortransform=priortransform, loglikelihood=loglikelihood, live_pointsu=live_pointsu, live_pointsx=live_pointsx, live_pointsL=live_pointsL, ndim=ndim, **kwargs)
		
		while True:
			# choose a new start point
			# can not be a phantom point, because those are below the threshold
			
			# work in whitened space
			startw = self.friends.metric.transform(startu)
			live_pointsw = self.friends.metric.transform(live_pointsu)
			def whitened_priortransform(wpoint):
				return priortransform(self.friends.metric.untransform(wpoint))
			def whitened_is_inside(wpoint):
				upoint = self.friends.metric.untransform(wpoint)
				return self.is_inside(upoint)
			
			w, x, L, n = self.mcmc_proposer.draw_constrained(Lmin, whitened_priortransform, loglikelihood, live_pointsw, ndim, self.friends.region.members, whitened_is_inside, startw, startx, startL)
			u = self.friends.metric.untransform(w)
			ntoaccept += n
			
			if self.is_inside(u) and L > Lmin:
				return u, x, L, ntoaccept
			
			# unsuccessful search so far. Try a different start point.
			starti = numpy.random.randint(len(live_pointsu))
			startu = live_pointsu[starti]
			startx = live_pointsx[starti]
			startL = live_pointsL[starti]
			
			# if running very inefficient, optimize clustering 
			#     if we haven't done so at the start
			if not rebuild and ntoaccept > 1000:
				rebuild = True
				self.friends.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=False)

	def stats(self):
		print('HybridFriendsConstrainer: %s' % ('used direct draws throughout (no MCMC)' if self.use_direct_draw else 'switched over to MCMC'))
		return self.mcmc_proposer.stats()

class HybridMultiEllipsoidConstrainer(object):
	"""
	Do MCMC within MultiEllipsoid constraints
	No metric learning.
	"""
	def __init__(self, mcmc_proposer, enlarge=1.2, update_interval=50, switchover_efficiency=0):
		self.enlarge = enlarge
		self.iter = -1
		self.update_interval = update_interval
		self.mcmc_proposer = mcmc_proposer
		self.switchover_efficiency = switchover_efficiency
		self.use_direct_draw = switchover_efficiency > 0
	
	def update(self, points):
		pointvol = exp(-self.iter / len(points)) / len(points)
		self.ells = bounding_ellipsoids(numpy.asarray(points), pointvol=pointvol)
		for ell in self.ells:
			ell.scale_to_vol(ell.vol * self.enlarge)
	
	def is_inside(self, u):
		if not (numpy.all(u > 0.) and numpy.all(u < 1.)):
			return False
		return any((ell.contains(u) for ell in self.ells))
	
	"""
	Return the found point with its likelihood, and the number of 
	likelihood calls performed.
	"""
	def draw_constrained(self, Lmin, priortransform, loglikelihood, live_pointsu, live_pointsx, live_pointsL, ndim, startu, startx, startL, **kwargs):
		self.iter += 1
		rebuild = False
		if self.iter % self.update_interval == 0:
			self.update(live_pointsu)
			rebuild = True
		ntoaccept = 0

		if self.use_direct_draw:
			while True:
				u = sample_ellipsoids(self.ells, rstate=numpy.random)
				if not (numpy.all(u > 0.) and numpy.all(u < 1.)):
					# try to draw from unit cube
					u = numpy.random.uniform(size=ndim)
					if not self.is_inside(u):
						# also did not work
						continue
				x = priortransform(u)
				L = loglikelihood(x)
				ntoaccept = ntoaccept + 1
				if Lmin is None or L > Lmin:
					return u, x, L, ntoaccept
				if 1. / ntoaccept < self.switchover_efficiency:
					print('low efficiency triggered switching over to MCMC')
					self.use_direct_draw = False
					break
		while True:
			# choose a new start point
			# can not be a phantom point, because those are below the threshold
			u, x, L, n = self.mcmc_proposer.draw_constrained(Lmin, priortransform, loglikelihood, live_pointsu, ndim, live_pointsu, self.is_inside, startu, startx, startL)
			ntoaccept += n
			
			if self.is_inside(u) and L > Lmin:
				return u, x, L, ntoaccept
			
			# unsuccessful search so far. Try a different start point.
			starti = numpy.random.randint(len(live_pointsu))
			startu = live_pointsu[starti]
			startx = live_pointsx[starti]
			startL = live_pointsL[starti]
			
			# if running very inefficient, optimize clustering 
			#     if we haven't done so at the start
			if not rebuild and ntoaccept > 1000:
				rebuild = True
				self.update(live_pointsu)

	def stats(self):
		print('HybridFriendsConstrainer: %s' % ('used direct draws throughout (no MCMC)' if self.use_direct_draw else 'switched over to MCMC'))
		return self.mcmc_proposer.stats()

from nested_sampling.clustering.sdml import IdentityMetric, SimpleScaling, TruncatedScaling, MahalanobisMetric, TruncatedMahalanobisMetric, SDML, TruncatedSDML

class HybridMLMultiEllipsoidConstrainer(object):
	"""
	Do MCMC within MultiEllipsoid constraints; 
	Metric-learn from ellipsoid points.
	
	This is EAGLENEST-fast.
	
	mcmc_proposer: 
	
	metriclearner: 'none', 'simplescaling' or 'sdml'
	
	enlarge: Volume enlarging factor.
	
	update_interval: Re-cluster every x iterations.
	
	switchover_efficiency: if 0, always use MCMC. 
	    Otherwise, use direct draws from multiellipsoids. If this becomes
	    inefficient (rejection rate below switchover_efficiency),
	    use MCMC for the remainder of the run.
	
	"""
	def __init__(self, mcmc_proposer, metriclearner, enlarge=1.2, 
		update_interval=50, switchover_efficiency=0,
		bs_enabled=False, bs_rounds=10, bs_memory=20):
		self.enlarge = enlarge
		self.iter = -1
		self.update_interval = update_interval
		self.mcmc_proposer = mcmc_proposer
		self.metriclearner = metriclearner
		self.switchover_efficiency = switchover_efficiency
		self.use_direct_draw = switchover_efficiency > 0
		self.metric = IdentityMetric()
		self.bs_enabled = bs_enabled
		self.bs_rounds = bs_rounds
		self.bs_memory = bs_memory
		self.lastdistmax = []
	
	def update(self, points):
		points = numpy.asarray(points)
		pointvol = exp(-self.iter / len(points)) / len(points)
		npoints, ndim = points.shape
		if self.bs_enabled:
			# bootstrapping rounds:
			for i in range(self.bs_rounds):
				choice = set(numpy.random.choice(numpy.arange(npoints), size=npoints))
				mask = numpy.array([c in choice for c in numpy.arange(npoints)])
				points_without_i = points[mask,:]
				ells = bounding_ellipsoids(points_without_i, pointvol=pointvol)
				# check distance of left-out points
				distmax = 1
				for i in numpy.where(~mask)[0]:
					dists = []
					for ell in ells:
						x = points[i,:] - ell.ctr
						d = numpy.dot(numpy.dot(x, ell.a), x)
						dists.append(d)
					distmax = max(distmax, min(dists))
				self.lastdistmax.append(distmax)
				if len(self.lastdistmax) > self.bs_rounds:
					break
			# remember back 50 steps
			enlarge = max(self.lastdistmax[-self.bs_memory:])
			enlarge = max(1, min(100, enlarge))
			enlarge = enlarge * ndim**0.5
			print('enlargement factor: %.2f' % enlarge, 'memory: ', ' '.join(['%.2f' % d for d in self.lastdistmax[-self.bs_memory:]]))
		else:
			enlarge = self.enlarge

		self.ells = bounding_ellipsoids(points, pointvol=pointvol)
		for ell in self.ells:
			ell.scale_to_vol(ell.vol * enlarge)
		
		# now update the metric. 
		# For this we need the individual clusters
		# Ideally we would find clusters by merging the overlapping 
		# ellipsoids. Unfortunately there is no analytic solution to
		# finding whether two n-ellipsoids overlap.
		# Instead, we check for points which live in more than one 
		# ellipsoid
		
		# for each point, find all ellipses that constain it
		ell_redirect = {i:i for i in range(len(self.ells))}
		ell_indices = []
		for u in points:
			ell_containing = None
			for i, ell in enumerate(self.ells):
				if ell.contains(u):
					if ell_containing is None:
						ell_containing = ell_redirect[i]
					else:
						ell_redirect[i] = ell_containing
			ell_indices.append(ell_containing)
		
		
		ell_indices = numpy.asarray([ell_redirect[i] for i in ell_indices])
		shifted_cluster_members = numpy.zeros_like(points)
		for i in set(ell_indices):
			mask = ell_indices == i
			clusterpoints = points[mask,:]
			mean = numpy.mean(clusterpoints, axis=0)
			shifted_cluster_members[mask] = clusterpoints - mean
		
		# construct whitening matrix
		metric_updated = False
		if self.metriclearner == 'none':
			metric = self.metric # stay with identity matrix
		elif self.metriclearner == 'simplescaling' or (self.metriclearner == 'mahalanobis' and ndim == 1):
			metric = SimpleScaling()
			metric.fit(shifted_cluster_members)
			metric_updated = True
		elif self.metriclearner == 'mahalanobis':
			metric = MahalanobisMetric()
			metric.fit(shifted_cluster_members)
			metric_updated = True
		elif self.metriclearner == 'sdml':
			metric = SDML()
			metric.fit(shifted_cluster_members, W = numpy.ones((len(shifted_cluster_members), len(shifted_cluster_members))))
			metric_updated = True
		else:
			assert False, self.metriclearner

		self.metric = metric
	
	def is_inside(self, u):
		if not (numpy.all(u > 0.) and numpy.all(u < 1.)):
			return False
		return any((ell.contains(u) for ell in self.ells))
	
	"""
	Return the found point with its likelihood, and the number of 
	likelihood calls performed.
	"""
	def draw_constrained(self, Lmin, priortransform, loglikelihood, live_pointsu, live_pointsx, live_pointsL, ndim, startu, startx, startL, **kwargs):
		self.iter += 1
		rebuild = False
		if self.iter % self.update_interval == 0:
			self.update(live_pointsu)
			rebuild = True
		ntoaccept = 0

		if self.use_direct_draw:
			while True:
				u = sample_ellipsoids(self.ells, rstate=numpy.random)
				if not (numpy.all(u > 0.) and numpy.all(u < 1.)):
					# try to draw from unit cube
					u = numpy.random.uniform(size=ndim)
					if not self.is_inside(u):
						# also did not work
						continue
				x = priortransform(u)
				L = loglikelihood(x)
				ntoaccept = ntoaccept + 1
				if Lmin is None or L > Lmin:
					return u, x, L, ntoaccept
				if 1. / ntoaccept < self.switchover_efficiency:
					print('low efficiency triggered switching over to MCMC')
					self.use_direct_draw = False
					break

		while True:
			# choose a new start point
			# can not be a phantom point, because those are below the threshold
			
			# work in whitened space
			startw = self.metric.transform(startu)
			live_pointsw = self.metric.transform(live_pointsu)
			def whitened_priortransform(wpoint):
				return priortransform(self.metric.untransform(wpoint))
			def whitened_is_inside(wpoint):
				upoint = self.metric.untransform(wpoint)
				return self.is_inside(upoint)
			
			w, x, L, n = self.mcmc_proposer.draw_constrained(Lmin, whitened_priortransform, loglikelihood, live_pointsw, ndim, live_pointsw, whitened_is_inside, startw, startx, startL)
			u = self.metric.untransform(w)
			ntoaccept += n
			
			if self.is_inside(u) and L > Lmin:
				return u, x, L, ntoaccept
			
			# unsuccessful search so far. Try a different start point.
			starti = numpy.random.randint(len(live_pointsu))
			startu = live_pointsu[starti]
			startx = live_pointsx[starti]
			startL = live_pointsL[starti]
			
			# if running very inefficient, optimize clustering 
			#     if we haven't done so at the start
			if not rebuild and ntoaccept > 1000:
				rebuild = True
				self.update(live_pointsu)

	def stats(self):
		print('HybridFriendsConstrainer: %s' % ('used direct draws throughout (no MCMC)' if self.use_direct_draw else 'switched over to MCMC'))
		return self.mcmc_proposer.stats()

