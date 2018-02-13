from __future__ import print_function
import scipy, scipy.stats
import numpy
from numpy import exp, log, log10, pi, cos, sin, dot, vdot
import numpy
from numpy.linalg import norm
from matplotlib import pyplot as plt
from nested_sampling.clustering.neighbors import find_maxdistance, find_rdistance, initial_rdistance_guess, nearest_rdistance_guess
import operator
getfirst = operator.itemgetter(0)

def random_unit_vector(ndim):
	v = numpy.random.normal(0, 1, size=ndim)
	v /= scipy.linalg.norm(v)
	return v

class HybridRadFriendsConstrainer(object):
	"""
	Base class to use RadFriends (with enforced shrinking, JackKnife resampling) 
	in combination with local step sampling methods.
	"""
	def __init__(self, nsteps = 20, plot = False):
		self.sampler = None
		self.phase = 0
		self.ndirect = 0
		self.ndirect_accepts = 0
		self.nfriends = 0
		self.nfriends_accepts = 0
		self.nsteps = nsteps
		self.region = None
		self.plot = plot

	def is_inside(self, u):
		"""
		Check if this new point is near or inside one of our clusters
		"""
		ndim = len(u)
		ulow = self.region['ulow']
		uhigh = self.region['uhigh']
		if not ((ulow <= u).all() and (uhigh >= u).all()):
			# does not even lie in our primitive rectangle
			# do not even need to compute the distances
			return False
		
		members = self.region['members']
		maxdistance = self.region['maxdistance']
		
		# if not initialized: no prefiltering
		if maxdistance is None:
			return True
		
		# compute distance to each member in each dimension
		dists = scipy.spatial.distance.cdist(members, [u], metric='euclidean')
		dist_criterion = dists < maxdistance
		# is it true for at least one?
		closeby = dist_criterion.any()
		return closeby

	def are_inside_rect(self, u):
		"""
		Check if the new points are near or inside one of our clusters
		"""
		ulow = self.region['ulow']
		uhigh = self.region['uhigh']
		mask = numpy.logical_and(((ulow <= u).all(axis=1), (uhigh >= u).all(axis=1)))
	def are_inside_cluster(self, us, ndim):
		members = self.region['members']
		maxdistance = self.region['maxdistance']
		
		dists = scipy.spatial.distance.cdist(members, us, metric='euclidean')
		dist_criterion = dists < maxdistance
		# is it true for at least one?
		closeby = dist_criterion.any(axis=0)
		return closeby
	
	def adapt(self):
		pass

	def local_step_sample(self, i, u1, x1, L1, Lmin, priortransform, loglikelihood, u):
		assert L1 >= Lmin
		maxdistance = self.region['maxdistance']
		k = 0
		# draw new random velocity
		if hasattr(self, 'velocities'):
			self.velocities[i] = random_unit_vector(ndim)
		
		if self.plot > 0:
			plt.figure(i)
			plt.gca().add_artist(plt.Circle((u1[0], u1[1]), maxdistance, color='grey'))
			plt.plot(u[:,0], u[:,1], 'o', color='grey')
			x = [u1[0]]
			y = [u1[1]]
		for j in range(self.nsteps):
			u2, x2, L2, k2 = self.step(i, u1, x1, L1, Lmin, priortransform, loglikelihood)
			k += k2
			u1, x1, L1 = u2, x2, L2
			if self.plot > 0:
				x.append(u1[0])
				y.append(u1[1])
		if self.plot > 0:
			plt.plot(x, y, 'x-', color='g')
			plt.gca().set_aspect(aspect='equal', adjustable='datalim')
			if i < 15:
				plt.savefig('galilean_%03d.pdf' % i)
			plt.close()
		return u1, x1, L1, k

	def generate_direct(self, ndim):
		# draw directly from prior
		ntotal = 0
		maxdistance = self.region['maxdistance']
		assert maxdistance is not None
		members = self.region['members']
		while True:
			us = numpy.random.uniform(self.region['ulow'], self.region['uhigh'], size=(100, ndim))
			ntotal += 100
			self.ndirect += 100
			mask = self.are_inside_cluster(us, ndim)
			self.ndirect_accepts += mask.sum()
			if not mask.any():
				continue
			us = us[mask]
			for u in us:
				yield u, ntotal
				ntotal = 0
	
	def generate_from_friends(self, ndim):
		# for small regions draw from points
		ntotal = 0
		maxdistance = self.region['maxdistance']
		assert maxdistance is not None
		members = self.region['members']
		N = 400
		while True:
			# choose random friend
			us = members[numpy.random.randint(0, len(members), N),:]
			ntotal += 100
			# draw direction around it
			direction = numpy.random.normal(0, 1, size=(N, ndim))
			direction = direction / ((direction**2).sum(axis=1)**0.5).reshape((-1,1))
			# choose radius: volume gets larger towards the outside
			# so give the correct weight with dimensionality
			radius = maxdistance * numpy.random.uniform(0, 1, size=(N,1))**(1./ndim)
			us = us + direction * radius
			inside = numpy.logical_and((us >= 0).all(axis=1), (us <= 1).all(axis=1))
			if not inside.any():
				continue
			us = us[inside]
			# count the number of points this is close to
			dists = scipy.spatial.distance.cdist(members, us, metric='euclidean')
			nnear = (dists < maxdistance).sum(axis=0)
			# accept with probability 1./nnear
			coin = numpy.random.uniform(size=len(us))
			accept = coin < 1. / nnear
			us = us[accept]
			for u in us:
				yield u, ntotal
				ntotal = 0

	def draw_constrained(self, Lmin, priortransform, loglikelihood, ndim, 
			previous, startu, startx, startL, starti, **kwargs):
		# compute RadFriends spheres
		u = numpy.array([u for u, _, L in previous if L >= Lmin])
		rebuilt = False
		if self.region is None or len(previous) % 50 == 0:
			# jackknife, is fast
			maxdistance = nearest_rdistance_guess(u, metric='euclidean')
			#maxdistance = find_rdistance(u, nbootstraps=50, metric='euclidean', verbose=False)
			# make sure we only shrink
			if self.region is not None and 'maxdistance' in self.region:
				maxdistance = min(maxdistance, self.region['maxdistance'])
			#print 'new distance:', maxdistance
			# compute enclosing rectangle for quick checks
			ulow  = numpy.max([u.min(axis=0) - maxdistance, numpy.zeros(ndim)], axis=0)
			uhigh = numpy.min([u.max(axis=0) + maxdistance, numpy.ones(ndim)], axis=0)
			self.region = dict(members=u, maxdistance=maxdistance, ulow=ulow, uhigh=uhigh)
			
			self.direct_generator = self.generate_direct(ndim=ndim)
			self.friends_generator = self.generate_from_friends(ndim=ndim)
			rebuilt = True
		
		ntoaccept = 0
		if self.phase == 0:
			# draw from rectangle until sphere rejection drops below 1%
			for u, ntotal in self.direct_generator:
				x = priortransform(u)
				L = loglikelihood(x)
				self.nfriends += 1
				ntoaccept += 1
				if L >= Lmin:
					if ntotal >= 200:
						print('Drawing from prior is becoming inefficient: %d draws before accept' % ntotal)
						print()
					if ntoaccept >= 200:
						print('RadFriends is becoming inefficient: %d draws until accept' % ntoaccept)
						print()
					self.nfriends_accepts += 1
					return u, x, L, ntoaccept
				if ntotal >= 1000:
					# drawing directly from prior 
					# becomes inefficient as we go to
					# small region
					# switch to drawing from Friends
					print('switching to Friends sampling phase')
					print()
					self.phase = 1
					break
				if ntoaccept >= 2000:
					print('switching to local steps sampling phase')
					print()
					self.phase = 2
					# drawing using RadFriends can become
					# inefficient in high dimensionality
					# switch to local step sampling
					break
		if self.phase == 1:
			# draw from spheres until acceptance rate drops below 0.05%
			for u, ntotal in self.friends_generator:
				x = priortransform(u)
				L = loglikelihood(x)
				self.nfriends += 1
				ntoaccept += 1
				if L >= Lmin:
					if ntoaccept >= 200:
						print('RadFriends is becoming inefficient: %d draws until accept' % ntoaccept)
						print()
					self.nfriends_accepts += 1
					return u, x, L, ntoaccept
				if ntoaccept >= 2000:
					print( 'switching to local steps sampling phase')
					print()
					self.phase = 2
					# drawing using RadFriends can become
					# inefficient in high dimensionality
					# switch to local step sampling
					break
		#print 'falling through...'
		# then do local step sampling
		i = starti # particle ID
		u1, x1, L1, k = self.local_step_sample(i, startu, startx, startL, 
			Lmin, priortransform, loglikelihood, u)
		#self.stats()
		self.adapt()
		return u1, x1, L1, k + ntoaccept
	
	def stats(self):
		pass


class GalileanRadFriendsConstrainer(HybridRadFriendsConstrainer):
	"""
	Galilean Nested Sampling is a special case of Nested Sampling using Hamiltonian Monte Carlo.
	Here we use RadFriends to determine the reflection surfaces.
	"""
	def __init__(self, nlive_points, ndim, velocity_scale = 0.5, nsteps = 20, plot = False):
		HybridRadFriendsConstrainer.__init__(self, nsteps = nsteps, plot = plot)
		self.velocity_scale = velocity_scale
		self.velocities = []
		# random direction initialisation
		for i in range(nlive_points):
			self.velocities.append(random_unit_vector(ndim))
		self.velocities = numpy.array(self.velocities)
		self.nproceed = 0
		self.nreflect = 0
		self.nreverse = 0
		self.nproceed_total = 0
		self.nreflect_total = 0
		self.nreverse_total = 0
	
	def reflect(self, u, v):
		""" Find reflection surface for line u + t*v, t>0 """
		# Find set of RadFriend spheres that lie on the line
		# compute line-sphere intersection with every point
		members = self.region['members']
		r = self.region['maxdistance']
		
		intervals = []
		for c in members:
			uc = u - c
			v2 = dot(v, v)
			uc2 = dot(uc, uc)
			r2 = dot(r, r)
			vuc = dot(v, uc)
			root = vuc**2 - v2 * (uc2 - r2)
			if root < 0:
				continue
			if self.plot > 3:
				plt.gca().add_artist(plt.Circle((c[0], c[1]), r, color='r', alpha=0.1))
			t1 = (-vuc - root**0.5) / v2
			t2 = (-vuc + root**0.5) / v2
			#if True:
			#	start = v * t1 + u
			#	end   = v * t2 + u
			#	print 'enclosing:', start, end
			#	plt.plot([start[0], end[0]], [start[1], end[1]], '-', color='orange', lw=2)
			if t1 <= t2:
				intervals.append((t1, t2, c))
			else:
				intervals.append((t2, t1, c))
		
		# with the list of intervals, compute the overlapping interval
		# from the start point to find the last sphere
		
		# sort intervals by starting value
		intervals.sort(key=getfirst)
		
		# start and end of interval
		A, B, C = (0, 0, c)
		for t1, t2, c in intervals:
			if t2 < A: continue # wrong direction
			if B < t1: break    # end of continuous interval.
			if B < t2:
				# extend interval
				B = t2
				C = c
		assert C is not None
		#print 'line interval:', A, B
		# now we found the final sphere, C
		# return the normalisation vector, which is between
		# the intersection point and the center of the sphere
		p = B * v + u
		if self.plot > 2:
			plt.gca().add_artist(plt.Circle((C[0], C[1]), r, color='b', alpha=0.2))
			plt.plot(p[0], p[1], '+', color='b')

		#if not numpy.allclose(C, p):
		n = C - p
		# normalise
		n = n / norm(n)
		v2 = v - 2 * n * dot(n, v)
		#else:
		#	# we are starting at the exact center of the sphere, 
		#	# and reflecting off the sphere
		#	# so we just go backwards, as the sphere is perpendicular
		#	# everywhere.
		#	print 'center of sphere reflection'
		#	v2 = -v
		if self.plot > 2:
			plt.plot([p[0], (v2+p)[0]], [p[1], (p+v2)[1]], '-', color='b', lw=3, alpha=0.2)
		
		return u + v, v2
		## if the reflected point is further outside than the 
		## original point u + v, use u + v.
		# not allowed by detailled balance!
		#if B > 1:
		#	return u + v, v2
		#else:
		#	return p, v2
	
	def step(self, i, u1, x1, L1, Lmin, priortransform, loglikelihood):
		# guess for stepsize
		dice = numpy.random.uniform(0, 1)
		timestep = 0.1 if dice < 0.3 else 1
		scale = self.region['maxdistance'] * self.velocity_scale * timestep
		#print 'stepsize:', stepsize, self.velocity_scale, maxdistance
		v = self.velocities[i] * scale
		# Start with (ui, v) where L(x1) is OK
		# go along
		u2 = u1 + v
		k = 0
		# check if inside
		inside_superset = self.is_inside(u2)
		inside = False
		if inside_superset:
			x2 = priortransform(u2)
			L2 = loglikelihood(x2)
			k = k + 1
			inside = L2 >= Lmin
		if inside:
			# if L2 is ok, proceed with u2, v
			self.nproceed += 1
			#print u1, v, ' -- proceed -->', u2, v
			return u2, x2, L2, k
		elif self.plot > 1:
			plt.plot(u2[0], u2[1], 's', color='r')
		# not inside.
		# we need to reflect.
		# find reflection surface
		ureflect, vreflect = self.reflect(u1, v)
		# go there from the outside point
		# here we use a point a bit closer in, the boundary point
		u3 = ureflect + vreflect
		
		inside_superset = self.is_inside(u3)
		inside = False
		if inside_superset:
			x3 = priortransform(u3)
			L3 = loglikelihood(x3)
			k = k + 1
			inside = L3 >= Lmin
		if inside:
			# update velocity, with mild randomisation
			#small_angle = pi / 180 / 10
			#r = numpy.random.normal(size=len(vreflect))
			#r /= norm(r)
			#vreflect = vreflect * cos(small_angle) + r * scale * sin(small_angle)
			self.velocities[i] = vreflect / scale
			self.nreflect += 1
			#print u1, v, ' -- reflect -->', u3, vreflect
			return u3, x3, L3, k
		elif self.plot > 1:
			plt.plot(u3[0], u3[1], '^', color='r')
		
		# reflected point is also not inside -- very bad.
		# try reversing from starting point
		small_angle = pi / 180
		r = numpy.random.normal(size=len(vreflect))
		r /= norm(r)
		v = v * cos(small_angle) + r * scale * sin(small_angle)
		self.velocities[i] = -v / scale
		#print u1, v, ' -- reflect -->', u3, '-- reverse -->', u1, -v
		self.nreverse += 1
		return u1, x1, L1, k
	
	def adapt(self):
		n = self.nproceed + self.nreflect + self.nreverse
		pr = self.nproceed * 1. / n
		N = len(self.velocities)
		rr = self.nreverse * 1. * N / n
		if pr < 0.6: # too many proceeds
			self.velocity_scale /= 1.1
			print('velocity' + ' '*50 + 'v')
		elif pr > 0.85:
			self.velocity_scale *= 1.1
			print('velocity' + ' '*50 + '^')
		else:
			if rr > 0.10: # too many reverse, decrease scale
				self.velocity_scale /= 1.01
			elif rr < 0.03: # few reverses, can increase scale
				self.velocity_scale *= 1.01
			# else: # just right
		self.nproceed_total += self.nproceed
		self.nreflect_total += self.nreflect
		self.nreverse_total += self.nreverse
		
		# reset
		self.nproceed = 0
		self.nreflect = 0
		self.nreverse = 0
	
	def stats(self):
		n = self.nproceed_total + self.nreflect_total + self.nreverse_total
		if n == 0: return
		velocity_scale = self.velocity_scale
		print('GalileanConstrainer stats: nsteps: %d, %.3f%% proceeds, %.3f%% reflects, %.5f%% reverses ' % (
			n, self.nproceed_total * 100. / n,  self.nreflect_total * 100. / n, self.nreverse_total * 100. / n))
		if self.nproceed_total * 1. / n < 0.5:
			print('velocity_scale = %s (too large!, too few proceeds)' % velocity_scale)
		elif self.nproceed_total * 1. / n > 0.75:
			print('velocity_scale = %s (too small!, too many proceeds)' % velocity_scale)
		else:
			print('velocity_scale = %s (good by proceeds)' % velocity_scale)
		N = len(self.velocities)
		if self.nreverse_total * 1. / n * N < 0.25:
			print('velocity_scale = %s (too large!, too few reverse)' % velocity_scale)
		elif self.nreverse_total * 1. / n * N > 5:
			print('velocity_scale = %s (too small!, too many reverse)' % velocity_scale)
		else:
			print('velocity_scale = %s (good by reverses)' % velocity_scale)

class MCMCRadFriendsConstrainer(HybridRadFriendsConstrainer):
	"""
	Markov Chain Monte Carlo sampling
	
	Here we use RadFriends to restrict the proposal distribution.
	"""
	def __init__(self, proposal_scale = 3, nsteps = 20, plot = False):
		HybridRadFriendsConstrainer.__init__(self, nsteps = nsteps, plot = plot)
		self.proposal_scale = proposal_scale
		self.naccepts = 0
		self.nrejects = 0
		self.naccepts_total = 0
		self.nrejects_total = 0
		self.nskip = 0
	
	def adapt(self):
		n = self.naccepts + self.nrejects
		proposal_scale = self.proposal_scale
		ar = self.naccepts * 100. / n
		boost = self.nskip * 100. / n
		if ar < 85:
			print('proposal scale %.2f acceptance rate: %.2f%% %50s' % (self.proposal_scale, ar, 'v'))
			self.proposal_scale /= 1.01
		elif ar > 95:
			print('proposal scale %.2f acceptance rate: %.2f%% %50s' % (self.proposal_scale, ar, '^'))
			self.proposal_scale *= 1.01
		self.naccepts_total += self.naccepts
		self.nrejects_total += self.nrejects
		self.naccepts = 0
		self.nrejects = 0
		self.nskip = 0
	
	def step(self, i, u1, x1, L1, Lmin, priortransform, loglikelihood):
		scale = self.region['maxdistance'] * self.proposal_scale
		# restricted proposal
		while True:
			step = numpy.random.normal(0, scale, size=len(u1))
			u2 = u1 + step
			inside_superset = self.is_inside(u2)
			if inside_superset:
				break
			else:
				self.nskip += 1
		# check if inside
		x2 = priortransform(u2)
		L2 = loglikelihood(x2)
		k = 1
		inside = L2 >= Lmin
		if inside:
			# if L2 is ok, accept the point
			self.naccepts += 1
			return u2, x2, L2, k
		elif self.plot > 1:
			plt.plot(u2[0], u2[1], 's', color='r')
		# not inside.
		self.nrejects += 1
		return u1, x1, L1, k
		
	def stats(self):
		n = self.naccepts_total + self.nrejects_total
		if n == 0: return
		proposal_scale = self.proposal_scale
		ar = self.naccepts_total * 100. / n
		boost = self.nskip * 100. / n
		print('MCMCConstrainer stats: nsteps: %d, %.3f%% accepts, %.3f%% skipped (%d)' % n, ar, boost, self.nskip)
		if ar < 85:
			print('proposal_scale = %s (too large!, too few accepts)' % proposal_scale)
		elif ar > 95:
			print('proposal_scale = %s (too small!, too many accepts)' % proposal_scale)
		else:
			print('proposal_scale = %s (good by accepts)' % proposal_scale)


if __name__ == '__main__':
	# gauss likelihood, sample at 1sigma
	ndim = 10
	Lmin = -1
	numpy.random.seed(1)
	def priortransform(u): return u
	def loglikelihood(x):  return -0.5 * (((x - 0.5)/0.3)**2).sum()
	startpoint = numpy.random.uniform(0, 1, size=ndim)
	
	live_points = []
	while len(live_points) < 100:
		p = numpy.random.uniform(0, 1, size=ndim)
		if loglikelihood(p) >= Lmin:
			live_points.append((p, p, Lmin))
	print('have live points')
	
	#constrainer = GalileanRadFriendsConstrainer(100, ndim, 
	#	velocity_scale = 1, nsteps = 200, plot=3)
	constrainer = MCMCRadFriendsConstrainer(proposal_scale = 1, nsteps = 40, plot=3)
	#constrainer.phase = 2
	for starti, (startu, startx, startL) in enumerate(live_points):
		u, x, L, k = constrainer.draw_constrained(Lmin=Lmin, 
			priortransform=priortransform, loglikelihood=loglikelihood, ndim=ndim, 
			startu=startu, startx=startx, startL=startL, starti=starti,
			previous=live_points)
		#print u, L
		#break
	
	constrainer.stats()
	

