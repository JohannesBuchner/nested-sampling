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

class TerminationCriterion(object):
	"""
	Classical nested sampling error, with contribution from nested sampling
	(through information H) and tail of live points (Monte carlo error).

	The steps between the lowest and highest likelihood is integrated.
	The choice where the step is done (at the lower, higher value or the mid point
	gives a lower, upper and medium estimate. The medium estimate is returned.
	The distance to the upper/lower (maximum) is used as a conservative estimate 
	of the uncertainty.
	"""
	def __init__(self, tolerance=0.5, maxRemainderFraction=0, plot=True):
		self.converged = False
		self.totalZ = numpy.nan
		self.totalZerr = numpy.nan
		self.remainderZ = numpy.nan
		self.remainderZerr = numpy.nan
		self.tolerance = tolerance
		self.maxRemainderFraction = maxRemainderFraction
		self.plot = plot
		self.plotdata = {}
	
	def compute(self, remainder, logV, globalLmax, logZ):
		L0 = remainder[-1][2]
		logLs = [Li - L0 for ui, xi, Li in remainder]
		
		"""
			      x---   4
			  x---       3
		      x---           2
		  x---               1
		  |   |   |   |   |


		  1 + 2 + 3 + 4
		  2 + 3 + 4 + 4
		  1 + 1 + 2 + 3
		
		# the positive edge is L2, L3, ... L-1, L-1
		# the average  edge is L1, L2, ... L-2, L-1
		# the negative edge is L1, L1, ... L-2, L-2
		"""

		Ls = numpy.exp(logLs)
		LsMax = Ls.copy()
		LsMax[-1] = numpy.exp(globalLmax - L0)
		Lmax = LsMax[1:].sum() + LsMax[-1]
		Lmin = Ls[:-1].sum() + Ls[0]
		
		logLmid = log(Ls.sum()) + L0
		logZmid = logaddexp(logZ, logV + logLmid)
		logZup  = logaddexp(logZ, logV + log(Lmax) + L0)
		logZlo  = logaddexp(logZ, logV + log(Lmin) + L0)
		logZerr = logZup - logZlo
		return logZmid, logV + logLmid, logZerr
	
	def update_converged(self):
		self.converged = self.totalZerr < self.tolerance
		if self.maxRemainderFraction > 0:
			self.converged = self.converged and (self.remainderZ - self.totalZ) < log(self.maxRemainderFraction)
		
	def update(self, sampler, logwidth, logVolremaining, logZ, H, globalLmax):
		remainder = list(sampler.remainder())
		logV = logwidth
		self.normalZ = logZ

		for i in range(len(remainder)):
			ui, xi, Li = remainder[i]
			wi = logwidth + Li
			logZnew = logaddexp(logZ, wi)
			H = exp(wi - logZnew) * Li + exp(logZ - logZnew) * (H + logZ) - logZnew
			logZ = logZnew
		
		self.normalZerr = (H / sampler.nlive_points)**0.5
		totalZ, remainderZ, remainderZerr = self.compute(remainder, logV, globalLmax, self.normalZ)
		self.remainderZ = remainderZ
		self.remainderZerr = remainderZerr
		self.remainderVolume = logV
		self.totalZ = totalZ
		self.totalZerr = self.normalZerr + self.remainderZerr
		self.update_converged()
		if self.plot: 
			self.update_plot()
	
	def update_plot(self):
		# record previous values
		for attr in ['remainderVolume', 'remainderZ', 'remainderZerr',
			'totalZ', 'totalZerr', 'normalZ', 'normalZerr']:
			self.plotdata[attr] = self.plotdata.get(attr, []) + [getattr(self, attr)]

class MaxErrorCriterion(TerminationCriterion):
	"""
	Conservative (over)estimation of remainder integral (namely, the live points). 
	The maximum/minimum likelihood is multiplied by the remaining volume 
	to give the highest/lowest Z.
	"""
	def compute(self, remainder, logV, globalLmax, logZ):
		L0 = remainder[-1][2]
		logLs = [Li - L0 for ui, xi, Li in remainder]
		Ls = numpy.exp(logLs)
		
		logLmid = log(Ls.sum()) + L0
		logZmid = logaddexp(logZ, logV + logLmid)
		logZmax = logaddexp(logZ, logV + globalLmax + log(len(logLs)))
		logZmin = logaddexp(logZ, logV + min(logLs) + log(len(logLs)))
		logZerr = logZmax - logZmin
		return logZmid, logV + logLmid, logZerr

class BootstrappedCriterion(TerminationCriterion):
	"""
	Bootstraps the live points for a more conservative error estimate.
	"""
	def compute(self, remainder, logV, globalLmax, logZ):
		L0 = remainder[-1][2]
		logLs = [Li - L0 for ui, xi, Li in remainder]
		
		Ls = numpy.exp(logLs)
		LsMax = Ls.copy()
		LsMax[-1] = numpy.exp(globalLmax - L0)
		Lmax = LsMax[1:].sum() + LsMax[-1]
		Lmin = Ls[:-1].sum() + Ls[0]
		logLmid = log(Ls.sum()) + L0
		logZmid = logaddexp(logZ, logV + logLmid)
		logZup  = logaddexp(logZ, logV + log(Lmax) + L0)
		logZlo  = logaddexp(logZ, logV + log(Lmin) + L0)
		logZerr = logZup - logZlo
		
		# try bootstrapping for error estimation
		bs_logZmids = []
		for _ in range(20):
			i = numpy.random.randint(0, len(Ls), len(Ls))
			i.sort()
			bs_Ls = LsMax[i]
			Lmax = bs_Ls[1:].sum() + bs_Ls[-1]
			bs_Ls = Ls[i]
			Lmin = bs_Ls[:-1].sum() + bs_Ls[0]
			bs_logZmids.append(logaddexp(logZ, logV + log(Lmax.sum()) + L0))
			bs_logZmids.append(logaddexp(logZ, logV + log(Lmin.sum()) + L0))
		bs_logZerr = numpy.max(bs_logZmids) - numpy.min(bs_logZmids)
		logZerr = max(bs_logZerr, logZerr)
		
		return logZmid, logV + logLmid, logZerr

class RememberingBootstrappedCriterion(BootstrappedCriterion):
	"""
	Remembers the variance in Lmax-Lmin and anticipates upcoming spikes.
	
	Memory is memory_length times longer than the average number of iterations
	between the last five Lmax changes.
	In other word, new Lmax is found at some iterations, it finds
	 the last five times that happens, computes the average duration between them.
	Then considers a memory of memory_length times longer than that, corresponding
	to approximately memory_length spikes.
	
	The mean of the loglikelihood bandwidth (Lmax - Lmin) is considered from 
	the memory. The value Lmin + bandwidth replaces the 
	maximum loglikelihood value in the error calculation
	
	Bootstraps the live points for a conservative error estimate.
	"""
	def __init__(self, memory_length=3., bumpingMinRemainder=0.1, **kwargs):
		BootstrappedCriterion.__init__(self, **kwargs)
		self.memory_length = memory_length
		self.memory_Lmax = []
		self.memory_Lmin = []
		self.bumpingMinRemainder = bumpingMinRemainder
	
	def compute(self, remainder, logV, globalLmax, logZ):
		L0 = remainder[-1][2]
		logLs = [Li for ui, xi, Li in remainder]
		
		logLmin = min(logLs)
		self.memory_Lmax.append(max(logLs))
		self.memory_Lmin.append(logLmin)
		spikes = numpy.where([Lprev != Lnext for Lprev, Lnext in zip(self.memory_Lmax[:-1], self.memory_Lmax[1:])])[0]
		#print 'memory: ', self.memory_Lmax
		
		if len(spikes) > 1 and self.memory_length > 0:
			lastspikes = spikes[-5:]
			spikedelta = numpy.mean([ihi - ilo for ilo, ihi in zip(lastspikes[:-1], lastspikes[1:])])
			nmemory = int(numpy.ceil(self.memory_length * spikedelta))
			bandwidth = numpy.mean([Lhi - Llo for Lhi, Llo in zip(self.memory_Lmax[-nmemory:], self.memory_Lmin[-nmemory:])])
			print('RememberingBootstrappedCriterion: memory considered: ', nmemory, lastspikes, bandwidth)
			
			if logLmin + bandwidth > globalLmax:
				print('RememberingBootstrappedCriterion: bumped Lmax', globalLmax, logLmin + bandwidth)
				globalLmax = logLmin + bandwidth
		totalZ, remainderZ, remainderZerr = BootstrappedCriterion.compute(self, remainder, logV, globalLmax, logZ)
		if False and self.bumpingMinRemainder > 0 and (remainderZ - totalZ) < log(self.bumpingMinRemainder):
			self.memory_length = 0
			print('RememberingBootstrappedCriterion: disabled bumping, remainder small')
		return totalZ, remainderZ, remainderZerr



class NoisyBootstrappedCriterion(TerminationCriterion):
	"""
	Bootstraps the live points for a conservative error estimate.
	
	When the normal (non-bootstrapped) error is below the tolerance,
	starts recording the evidence estimates.
	The standard deviation of the evidence since that time is added to the 
	uncertainty.
	
	Convergence is achieved when the bootstrapped error is below the 
	tolerance *and* the normal error + std are below the uncertainty.
	"""
	def __init__(self, conservative=False, **kwargs):
		TerminationCriterion.__init__(self, **kwargs)
		self.remainder_memory = []
		self.start_recording = False
		self.conservative = conservative
	
	def update_converged(self):
		self.converged = self.start_recording and self.totalZerr < self.tolerance
		if self.converged and self.maxRemainderFraction > 0:
			self.converged = self.converged and (self.remainderZ - self.totalZ) < log(self.maxRemainderFraction)
		if self.converged and (self.remainderZ - self.totalZ) > log(0.1):
			sigma = numpy.std(self.remainder_memory)
			if self.conservative:
				print('NoisyBootstrappedCriterion(conservative): %.1f scatter additional to %.1f, ratio %.3f' % (sigma, self.remainderZerr, exp(self.remainderZ - self.normalZ)))
				self.converged = (sigma + self.remainderZerr) < self.tolerance
			else:
				print('NoisyBootstrappedCriterion: %.1f scatter additional to %.1f, ratio %.3f' % (sigma, self.classic_totalZerr, exp(self.remainderZ - self.normalZ)))
				self.converged = (sigma**2 + self.classic_totalZerr**2)**0.5 < self.tolerance
			#self.converged = sigma < self.normalZerr

	def update_plot(self):
		TerminationCriterion.update_plot(self)
		attr = 'memory_sigma'
		self.plotdata[attr] = self.plotdata.get(attr, []) + [numpy.std(self.remainder_memory)]
	
	def compute(self, remainder, logV, globalLmax, logZ):
		L0 = remainder[-1][2]
		logLs = [Li - L0 for ui, xi, Li in remainder]
		
		Ls = numpy.exp(logLs)
		LsMax = Ls.copy()
		LsMax[-1] = numpy.exp(globalLmax - L0)
		Lmax = LsMax[1:].sum() + LsMax[-1]
		Lmin = Ls[:-1].sum() + Ls[0]
		logLmid = log(Ls.sum()) + L0
		logZmid = logaddexp(logZ, logV + logLmid)
		logZup  = logaddexp(logZ, logV + log(Lmax) + L0)
		logZlo  = logaddexp(logZ, logV + log(Lmin) + L0)
		logZerr = logZup - logZlo
		
		# try bootstrapping for error estimation
		bs_logZmids = []
		for _ in range(20):
			i = numpy.random.randint(0, len(Ls), len(Ls))
			i.sort()
			bs_Ls = LsMax[i]
			Lmax = bs_Ls[1:].sum() + bs_Ls[-1]
			bs_Ls = Ls[i]
			Lmin = bs_Ls[:-1].sum() + bs_Ls[0]
			bs_logZmids.append(logaddexp(logZ, logV + log(Lmax.sum()) + L0))
			bs_logZmids.append(logaddexp(logZ, logV + log(Lmin.sum()) + L0))
		bs_logZerr = numpy.max(bs_logZmids) - numpy.min(bs_logZmids)
		
		totalZerr = self.normalZerr + logZerr
		self.classic_totalZerr = totalZerr
		if not self.start_recording and totalZerr < self.tolerance:
			print('NoisyBootstrappedCriterion: %.1f with err %.1f BS: %.1f' % (logV + logLmid, logZerr, bs_logZerr))
			self.start_recording = True
		if self.start_recording:
			self.remainder_memory.append(logV + logLmid)
			#sigma = numpy.std(self.remainder_memory)
			#print 'NoisyBootstrappedCriterion: %.1f scatter additional to %.1f' % (sigma, self.normalZerr)
			#logZerr = (sigma**2 + logZerr**2)**0.5
		
		logZerr = max(bs_logZerr, logZerr)
		return logZmid, logV + logLmid, logZerr



class NoiseDetectingBootstrappedCriterion(NoisyBootstrappedCriterion):
	"""
	Bootstraps the live points for a conservative error estimate.
	
	When the normal (non-bootstrapped) error is below the tolerance,
	starts recording the evidence estimates.
	The standard deviation of the evidence since that time is added to the 
	uncertainty.
	
	Convergence is achieved when the bootstrapped error is below the 
	tolerance *and* the normal error + std are below the uncertainty.
	"""
	def __init__(self, maxNoisyRemainder = 0.1, **kwargs):
		NoisyBootstrappedCriterion.__init__(self, **kwargs)
		self.maxNoisyRemainder = maxNoisyRemainder
	
	def update_converged(self):
		self.converged = self.start_recording and self.totalZerr < self.tolerance
		if self.converged and self.maxRemainderFraction > 0:
			self.converged = self.converged and (self.remainderZ - self.totalZ) < log(self.maxRemainderFraction)
		if self.converged:
			sigma = numpy.std(self.remainder_memory)
			if sigma > self.normalZerr:
				print('NoiseDetectingBootstrappedCriterion: %.1f scatter additional to %.1f, remainder fraction %.3f' % (sigma, self.remainderZerr, exp(self.remainderZ - self.totalZ)))
				self.converged = (self.remainderZ - self.totalZ) > log(self.maxNoisyRemainder)

class DecliningBootstrappedCriterion(TerminationCriterion):
	"""
	Bootstraps the live points for a conservative error estimate.
	
	When the normal (non-bootstrapped) error is below the tolerance,
	starts recording the evidence estimates.
	At later times, the standard deviation of the evidence is computed since
	that time, and the decline since then.
	For convergence, a logarithmic decrease of 1 plus a standard deviation
	is required.
	"""
	def __init__(self, required_decrease=1., required_decrease_scatter=1., **kwargs):
		TerminationCriterion.__init__(self, **kwargs)
		self.required_decrease_scatter = required_decrease_scatter
		self.required_decrease = required_decrease
		self.remainder_memory = []
		self.start_recording = False
	
	def update_converged(self):
		self.converged = self.start_recording and self.totalZerr < self.tolerance
		if self.converged and self.maxRemainderFraction > 0:
			self.converged = self.converged and (self.remainderZ - self.totalZ) < log(self.maxRemainderFraction)
		if self.converged and (self.remainderZ - self.totalZ) > log(0.1):
			sigma = numpy.std(self.remainder_memory)
			threshold = self.remainder_memory[0] - self.required_decrease_scatter * sigma - self.required_decrease
			print('DecliningBootstrappedCriterion: %.1f (need <%.1f because of %.1f scatter, ratio: %.3f)' % (self.remainder_memory[-1], threshold, sigma, exp(self.remainderZ - self.normalZ)))
			self.converged = self.remainder_memory[-1] < threshold
	
	def compute(self, remainder, logV, globalLmax, logZ):
		L0 = remainder[-1][2]
		logLs = [Li - L0 for ui, xi, Li in remainder]
		
		Ls = numpy.exp(logLs)
		LsMax = Ls.copy()
		LsMax[-1] = numpy.exp(globalLmax - L0)
		Lmax = LsMax[1:].sum() + LsMax[-1]
		Lmin = Ls[:-1].sum() + Ls[0]
		logLmid = log(Ls.sum()) + L0
		logZmid = logaddexp(logZ, logV + logLmid)
		logZup  = logaddexp(logZ, logV + log(Lmax) + L0)
		logZlo  = logaddexp(logZ, logV + log(Lmin) + L0)
		logZerr = logZup - logZlo
		
		# try bootstrapping for error estimation
		bs_logZmids = []
		for _ in range(20):
			i = numpy.random.randint(0, len(Ls), len(Ls))
			i.sort()
			bs_Ls = LsMax[i]
			Lmax = bs_Ls[1:].sum() + bs_Ls[-1]
			bs_Ls = Ls[i]
			Lmin = bs_Ls[:-1].sum() + bs_Ls[0]
			bs_logZmids.append(logaddexp(logZ, logV + log(Lmax.sum()) + L0))
			bs_logZmids.append(logaddexp(logZ, logV + log(Lmin.sum()) + L0))
		bs_logZerr = numpy.max(bs_logZmids) - numpy.min(bs_logZmids)
		
		totalZerr = self.normalZerr + logZerr
		if not self.start_recording and totalZerr < self.tolerance:
			print('DecliningBootstrappedCriterion: %.1f with err %.1f BS: %.1f' % (logV + logLmid, logZerr, bs_logZerr))
			self.start_recording = True
		if self.start_recording:
			self.remainder_memory.append(logV + logLmid)
		
		logZerr = max(bs_logZerr, logZerr)
		return logZmid, logV + logLmid, logZerr


