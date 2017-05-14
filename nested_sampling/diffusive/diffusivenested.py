"""
Pure Python implementation of Diffusive Nested Sampling.
"""

import numpy
from numpy import exp, log, log10, pi

class Level(object):
	def __init__(self, logX, cutoff, accepts=0, tries=0, visits=0, exceeds=0):
		self.cutoff = cutoff
		self.logX = logX

		self.accepts = accepts
		self.tries = tries
		self.visits = visits
		self.exceeds = exceeds
	def renormaliseVisits(self, regularisation):
		if self.tries >= regularisation:
			self.accepts = ((self.accepts + 1.)/(self.tries + 1.)) * regularisation
			self.tries = regularisation
		if self.visits >= regularisation:
			self.exceeds = ((self.exceeds + 1.)/(self.visits + 1.)) * regularisation
			self.visits = regularisation
	def incrementVisits(self, incrementExceeds):
		self.visits += 1
		if incrementExceeds:
			self.exceeds += 1
	def incrementTries(self, accepted):
		self.tries += 1
		if accepted:
			self.accepts += 1
	def __add__(self, other):
		assert self.logX == other.logX and self.cutoff == other.cutoff
		self.tries += other.tries
		self.accepts += other.accepts
		self.visits += other.visits
		self.exceeds += other.exceeds
	def __sub__(self, other):
		assert self.logX == other.logX and self.cutoff == other.cutoff
		self.tries -= other.tries
		self.accepts -= other.accepts
		self.visits -= other.visits
		self.exceeds -= other.exceeds
	def __lt__(self, other):
		raise NotImplementedError()
	def __gt__(self, other):
		raise NotImplementedError()
	def __eq__(self, other):
		raise NotImplementedError()
	def __ne__(self, other):
		raise NotImplementedError()
	def __mul__(self, other):
		raise NotImplementedError()
	def __div__(self, other):
		raise NotImplementedError()
	
def recalculateLogX(levels, compression, regularisation):
	assert len(levels) > 0
	levels[0].logX = 0.
	lastlevel = levels[0]
	for l in levels[1:]:
		f = log((lastlevel.exceeds + regularisation*1. / compression) / (lastlevel.visits + regularisation))
		l.logX = lastlevel.logX + f
		lastlevel = l
def renormaliseVisits(levels, regularisation):
	for l in levels:
		l.renormaliseVisits(regularisation)

def binaryCoin():
	return numpy.random.randint(2) == 0

def randh():
	return 10**(1.5 - 6 * numpy.random.uniform())*numpy.random.normal()

class DiffusiveSampler(object):
	def __init__(self, priortransform, loglikelihood, draw_constrained, 
		ndim,
		nlive_points = 1, # number of particles
		compression = numpy.e, 
		histogramForce = 5, # beta
		maxLevels = 5, 
		newLevelInterval=100, 
		backtrackScale = 10, # lambda
		#numParticles = 1, 
		stepInterval = 100 # number of iterations before returning as sample
	):
		
		self.nlive_points = nlive_points
		self.beta = histogramForce
		self.lam = backtrackScale
		self.compression = compression
		self.priortransform = priortransform
		self.loglikelihood = loglikelihood
		self.draw_constrained = draw_constrained
		self.maxLevels = maxLevels
		self.newLevelInterval = newLevelInterval
		self.saveInterval = stepInterval # store after this many steps
		self.levels = [Level(logX=0., cutoff=-1e300)]
		self.indices = [] # which level a point belongs to
		self.samples = []
		self.keep = [] # loglikelihoods which are not accepted
		self.ndim = ndim
		# draw N starting points from prior
		live_pointsu = [None] * nlive_points # particles
		live_pointsx = [None] * nlive_points
		live_pointsL = numpy.empty(nlive_points) # logL
		for i in range(nlive_points):
			u = numpy.random.uniform(0, 1, size=ndim)
			assert len(u) == ndim, (u, ndim)
			x = priortransform(u)
			assert len(x) == ndim, (x, ndim)
			L = loglikelihood(x)
			live_pointsu[i], live_pointsx[i], live_pointsL[i] = u, x, L
			self.samples.append([u, x, L])
			self.keep.append(L)
			self.indices.append(0)
		self.live_pointsu = live_pointsu
		self.live_pointsx = live_pointsx
		self.live_pointsL = live_pointsL
		self.Lmax = self.live_pointsL.max()
		self.neval = 0
	
	def levelsComplete(self):
		return self.maxLevels > 0 and len(self.levels) == self.maxLevels
	
	def __next__(self):
		for i in range(self.saveInterval):
			# choose random particle to move
			which = numpy.random.randint(self.nlive_points)
			if binaryCoin():
				self.updateParticle(which)
				self.updateIndex(which)
			else:
				self.updateIndex(which)
				self.updateParticle(which)
			# Accumulate visits, exceeds
			# move up in levels if necessary
			Li = self.live_pointsL[which]
			for index in range(self.indices[which], len(self.levels) - 1):
				exceeds = self.levels[index + 1].cutoff < Li
				self.levels[index].incrementVisits(exceeds)
				if not exceeds:
					break
			# Accumulate likelihoods for making a new level
			if not self.levelsComplete() and self.levels[-1].cutoff < Li:
				self.keep.append(Li)
			
			# bookKeeping:
			if len(self.keep) > self.newLevelInterval:
				self.keep.sort()
				ii = int((1. - 1. / self.compression) * len(self.keep))
				cutoff = self.keep[ii]
				print "# Creating level %d with logL = %.2f." % (len(self.levels), cutoff)
				self.levels.append(Level(self.levels[-1].logX - 1., cutoff))
				if self.levelsComplete():
					self.keep = []
					renormaliseVisits(self.levels, self.newLevelInterval)
				else:
					self.keep = self.keep[ii + 1:]
			
				recalculateLogX(self.levels, self.compression, 100)
				self.deleteParticle()
		
		print ''.join(['*' if (self.indices == i).any() else ' ' for i in range(len(self.levels))])
		
		sample = self.live_pointsu[which]
		sampleInfo = self.indices[which], self.live_pointsL[which], numpy.random.uniform(), which
		recalculateLogX(self.levels, self.compression, 100)
		
		return sample, sampleInfo

	def logPush(self, index):
		assert index >= 0 and index < len(self.levels)
		
		if self.levelsComplete():
			return 0
		i = index - (len(self.levels) - 1)
		return i * 1. / self.lam
	
	def deleteParticle(self):
		# Flag each particle as good or bad
		bad = [self.logPush(self.indices[i]) < -5. for i in range(self.nlive_points)]
		nbad = sum(bad)

		assert nbad < self.nlive_points, "# Warning: all particles lagging! Very rare!"
		# Replace bad particles with copies of good ones
		to_replace = [i for i, isbad in enumerate(bad) if isbad]
		for i in to_replace:
			copy = i
			while bad[copy]:
				copy = numpy.random.randint(self.nlive_points)
			self.live_pointsu[i], self.live_pointsx[i], self.live_pointsL[i] = \
				self.live_pointsu[copy], self.live_pointsx[copy], self.live_pointsL[copy]
			self.indices[i] = self.indices[copy]
			print "# Deleting a particle. Replacing it with a copy of a good survivor."
		
	def updateParticle(self, which):
		"""
		Move live point / particle in parameter space
		"""
		# Copy the particle
		# Perturb the proposal particle
		uj, xj, Lj = self.draw_constrained(
			Lmin = None, # we want to get the accept
			priortransform=self.priortransform, 
			loglikelihood=self.loglikelihood, 
			previous=self.samples,
			ndim=self.ndim,
			startu = self.live_pointsu[which], 
			startx = self.live_pointsx[which], 
			startL = self.live_pointsL[which],
			level=self.indices[which])
		
		self.Lmax = max(Lj, self.Lmax)
		self.samples.append([uj, xj, Lj])
		self.neval += 1
		
		accepted = self.levels[self.indices[which]].cutoff < Lj
		if accepted:
			# accept
			self.live_pointsu[which] = uj
			self.live_pointsx[which] = xj
			self.live_pointsL[which] = Lj
			#print 'draw accepted', Lj, self.levels[self.indices[which]].cutoff
		self.levels[self.indices[which]].incrementTries(accepted)
	
	def randomLevel_original(self, index, nLevels):
		u = (10.**(2*numpy.random.uniform()) * numpy.random.normal())
		offset = int(round(u))
		proposedIndex = index + offset
		
		if proposedIndex == index:
			proposedIndex = proposedIndex + 1 if binaryCoin() else proposedIndex - 1
		return proposedIndex

	def randomLevel_small(self, index, nLevels):
		u = (10.**(2*numpy.random.uniform()) * numpy.random.normal(0, 0.1))
		offset = int(round(u))
		proposedIndex = index + offset
	
		if proposedIndex == index:
			proposedIndex = proposedIndex + 1 if binaryCoin() else proposedIndex - 1
		return proposedIndex

	def randomLevel_manual(self, index, nLevels):
		coin = numpy.random.uniform()
		if coin < 0.01:
			offset = -4
		elif coin < 0.1:
			offset = -3
		elif coin < 0.2:
			offset = -2
		elif coin < 0.5:
			offset = -1
		elif coin < 0.8:
			offset = 1
		elif coin < 0.9:
			offset = 2
		elif coin < 0.99:
			offset = 3
		else:
			offset = 4
		proposedIndex = index + offset
		return proposedIndex
	
	def randomLevel(self, index, nLevels):
		return self.randomLevel_original(index, nLevels)
	
	def updateIndex(self, which):
		"""
		Switch live point / particle to another level randomly
		"""
		index = self.indices[which]
		proposedIndex = self.randomLevel(index, len(self.levels))
		
		if proposedIndex < 0 or proposedIndex >= len(self.levels):
			return
		
		if self.levels[proposedIndex].cutoff >= self.live_pointsL[which]:
			# can not ascend
			return
		
		# Acceptance probability. logX part
		logA = self.levels[index].logX - self.levels[proposedIndex].logX
		# Pushing up part
		logA += self.logPush(proposedIndex) - self.logPush(index)

		# Enforce uniform exploration part (if all levels exist)
		if self.levelsComplete():
			logA += self.beta * log((self.levels[index].tries + 1.) / (self.levels[proposedIndex].tries + 1.))
		# Prevent exponentiation of huge numbers
		logA = min(0, logA)
		
		if numpy.random.uniform() <= exp(logA):
			# Accept!
			self.indices[which] = proposedIndex
	
	def store(self, which):
		with open('sample_info.txt', 'w') as f:
			f.write("# index, logLikelihood, tieBreaker, ID.\n")
			numpy.savetxt(f, self.samplesInfo, fmt='%d %e %f %d')
		with open('sample.txt', 'w') as f:
			f.write("# Samples file. One sample per line.\n")
			numpy.savetxt(f, self.samples)
		levels = numpy.array([
			[l.logX, l.cutoff, numpy.random.uniform(), 
			l.accepts, l.tries, l.exceeds, l.visits] 
				for l in self.levels])
		with open('levels.txt', 'w') as f:
			f.write("# Samples file. One sample per line.\n")
			numpy.savetxt(f, levels, fmt='%e %e %f %d %d %d %d')
		
	def next(self):
		return self.__next__()
	def __iter__(self):
		while True: yield self.__next__()

def mcmc_draw(Lmin, priortransform, loglikelihood, ndim, 
		startu, startx, startL, **kwargs):
	while True:
		# just make a proposal (non-romantic, platonic only)
		u = numpy.copy(startu)
		i = numpy.random.randint(ndim)
		p = u[i]
		#p += 10**(1.5 - 6 * numpy.random.uniform(0,1)) * numpy.random.normal(0, 1);
		p += randh()
		# wrap around:
		u[i] = p - numpy.floor(p)
		#u[i] = wrap(p, 0, 1)
		assert u[i] >= 0, u[i]
		assert u[i] <= 1, u[i]
		
		x = priortransform(u)
		L = loglikelihood(u)
		#print 'MCMC: %f -> %f, like %s -> %f' % (startu[i], u[i], Lmin, L)
		if Lmin is None or L > Lmin:
			return u, x, L

import postprocess
#import progressbar
def diffusive_integrator(sampler, tolerance = 0.01, maxEvaluations=0, minEvaluations=0):
	# sample from sampler
	samples = [] # coordinates
	samplesInfo = [] # logX, likelihood, (tieBreaker), livePointID

	finfo = open('sample_info.txt', 'w')
	finfo.write("# index, logLikelihood, tieBreaker, ID.\n")
	
	#numpy.savetxt(f, self.samplesInfo, fmt='%d %e %f %d')
	fsample = open('sample.txt', 'w')
	fsample.write("# Samples file. One sample per line.\n")
	#numpy.savetxt(f, self.samples)
	
	logZs = []
	i = 1
	nlevelcomplete = None
	while True:
		# keep sampling
		sample, sampleInfo = sampler.next()
		#print 'new sample:', sample, sampleInfo

		# each sample file contains one line per live point / particle
		# sampleFile :: self.live_pointsu
		# sampleInfoFile :: self.indices, self.live_pointsL, (tieBreaker), ID?
		# levelFile :: logX, cutoff, (tieBreaker), accepts, tries, exceeds, visits

		# adding an extra number because postprocess can't deal with 1d data
		fsample.write(' '.join(['%e' % s for s in sample]) + "\n")
		fsample.flush()
		finfo.write("%d %e %f %d\n" % sampleInfo)
		finfo.flush()
		
		samples.append(sample)
		samplesInfo.append(sampleInfo)
		
		levels = numpy.array([[l.logX, l.cutoff, numpy.random.uniform(), l.accepts, l.tries, l.exceeds, l.visits] for l in sampler.levels])
		flevels = open('levels.txt', 'w')
		flevels.write("# logX, logLikelihood, tieBreaker, accepts, tries, exceeds, visits.\n")
		numpy.savetxt(flevels, levels, fmt='%f %e %f %d %d %d %d')
		flevels.close()
		
		if i % 20 == 0:
			# check if tolerance achieved already
			logZ, H, weights, logZerr, ESS = postprocess.postprocess(
				loaded=(levels, numpy.asarray(samplesInfo), numpy.asarray(samples)),
				save=False, plot=False, verbose=False,
				numResampleLogX=10)
			logZs.append(logZ)
			print 'logZ = %.3f +- %.3f +- %.3f | %.1f samples | iteration %d[warmup:%s,neval:%d]' % (logZ, logZerr, numpy.std(logZs[-10:]), ESS, i, nlevelcomplete, sampler.neval)
			
			if nlevelcomplete is None and ESS > 60 and sampler.maxLevels <= 0:
				sampler.maxLevels = len(sampler.levels)
			if nlevelcomplete is None and sampler.levelsComplete():
				nlevelcomplete = i
				print '# levels sufficient after %d iterations' % nlevelcomplete
			
			logZerr_total = (logZerr**2 + numpy.var(logZs[-10:]))**0.5
			converged = nlevelcomplete is not None and logZerr_total < tolerance and numpy.std(logZs[-10:]) * 10 < tolerance
			
			if (minEvaluations is not None and sampler.neval > minEvaluations) and (converged or (maxEvaluations > 0 and sampler.neval >= maxEvaluations)):
				# do some plotting
				postprocess.postprocess(
					loaded=(levels, numpy.asarray(samplesInfo), numpy.asarray(samples)),
					save=True, plot=True,
					numResampleLogX=1)
				break
		i = i + 1
	
	return dict(logZ=logZ, logZerr=logZerr, 
		samples=samples, points=[(p, numpy.median(w), si[0]) for p, si, w in zip(samples, samplesInfo, weights)], 
			information=H)

if __name__ == '__main__':
	import scipy.stats
	def priortransform(u):
		return u
	ndim = 2
	rv = scipy.stats.norm([0.654321]*ndim, [0.01]*ndim)
	def loglikelihood(x):
		#a = - 0.5 * ((x - 0.2)/0.05)**2 - 0.5 * log(2*pi*0.05**2)
		#b = - 0.5 * ((x - 0.7)/0.05)**2 - 0.5 * log(2*pi*0.05**2)
		#l = log(exp(a) + exp(b) + 0.01e-100)
		#print 'loglike:', x, l
		l = rv.logpdf(x).sum()
		return float(l)

	constrainer = mcmc_draw

	print 'preparing sampler'
	sampler = DiffusiveSampler(nlive_points = 5,
		priortransform=priortransform, loglikelihood=loglikelihood, 
		draw_constrained = constrainer, ndim=ndim,
		maxLevels=0)
	print 'running sampler'
	result = diffusive_integrator(tolerance=0.5, sampler=sampler, maxEvaluations=2000000)
	
	i = numpy.argmax([x + L for p, x, L in result['points']])
	#xi = result['points'][i][1]
	#print i, xi, 1.5 * xi
	


