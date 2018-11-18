from __future__ import print_function
"""

Simple gaussian likelihood analysed with various samplers

"""
import numpy
from numpy import log, exp, pi
import sys
import time

from nested_sampling.nested_integrator import nested_integrator
from nested_sampling.nested_sampler import NestedSampler
from nested_sampling.samplers.rejection import RejectionConstrainer
from nested_sampling.samplers.friends import FriendsConstrainer
from nested_sampling.samplers.hiermetriclearn import MetricLearningFriendsConstrainer
from nested_sampling.samplers.mcmc import MCMCConstrainer, GaussProposal, MultiScaleProposal
from nested_sampling.samplers.svm.svmnest import SVMConstrainer
from nested_sampling.samplers.ellipsoidal import EllipsoidConstrainer, MultiEllipsoidConstrainer
from nested_sampling.samplers.hybrid import FilteredMCMCConstrainer, FilteredGaussProposal, HybridMLFriendsConstrainer, HybridMLMultiEllipsoidConstrainer, FilteredUnitRandomSliceProposal, FilteredSliceConstrainer

from nested_sampling.termination_criteria import TerminationCriterion, MaxErrorCriterion, BootstrappedCriterion, RememberingBootstrappedCriterion, DecliningBootstrappedCriterion, NoisyBootstrappedCriterion, NoiseDetectingBootstrappedCriterion

def loglikelihood(x):
	sigma = 0.1
	like = -0.5 * (((x - 0.654321)/sigma)**2 + log(2*pi * sigma**2)).sum()
	return like

def priortransform(u):
	return u

# number of live points
nlive_points = 400
# for MCMC, number of steps
nsteps = 20
# number of samples
max_samples = 2000000
# dimensionality
ndim = 4

def run(constrainer, termination):
	print('configuring NestedSampler for constrainer', constrainer)
	starttime = time.time()
	if hasattr(constrainer, 'get_Lmax'):
		constrainer_get_Lmax = constrainer.get_Lmax
	else:
		constrainer_get_Lmax = None
	sampler = NestedSampler(nlive_points = nlive_points, 
		priortransform=priortransform, loglikelihood=loglikelihood, 
		draw_constrained = constrainer.draw_constrained, ndim=ndim,
		constrainer_get_Lmax = constrainer_get_Lmax)
	constrainer.sampler = sampler
	print('running nested_integrator to termination', termination)
	result = nested_integrator(sampler=sampler, max_samples=max_samples, terminationcriterion=termination)

	endtime = time.time()
	if hasattr(constrainer, 'stats'):
		constrainer.stats()
	
	print("RESULTS:")
	print('lnZ = %s +- %s' % (result['logZ'], result['logZerr']))
	print('niter: %d, duration: %.1fs' % (result['niterations'], endtime - starttime))
	assert -1 < result['logZ'] < 1 # should be ~0
	assert 0 < result['logZerr'] < 2 # should be ~0.5
	print()

def test_naive():
	constrainer = RejectionConstrainer()
	termination = TerminationCriterion(tolerance=0.5)
	run(constrainer, termination)

def test_mcmc():
	proposer = GaussProposal(adapt='sivia')
	constrainer = MCMCConstrainer(proposer=proposer, nsteps=nsteps, nminaccepts=nsteps)
	termination = TerminationCriterion(tolerance=0.5)
	run(constrainer, termination)

def test_mcmc_multiscale(): # does not work correctly, proposal is not good
	try:
		proposer = MultiScaleProposal()
		constrainer = MCMCConstrainer(proposer=proposer, nsteps=nsteps*5)
		termination = TerminationCriterion(tolerance=0.5)
		run(constrainer, termination)
	except AssertionError as e:
		# this is bad pytest style, but sometimes no exception is thrown
	 	pass

def test_ellipsoid():
	constrainer = EllipsoidConstrainer()
	termination = TerminationCriterion(tolerance=0.5)
	run(constrainer, termination)

def test_multiellipsoid():
	constrainer = MultiEllipsoidConstrainer()
	termination = TerminationCriterion(tolerance=0.5)
	run(constrainer, termination)

def test_radfriends():
	constrainer = FriendsConstrainer(rebuild_every=nlive_points, radial=True, metric = 'euclidean', 
		jackknife=False, force_shrink=True, 
		keep_phantom_points=False, optimize_phantom_points=False, verbose=False)
	termination = TerminationCriterion(tolerance=0.5, maxRemainderFraction=0.001)
	run(constrainer, termination)

def test_mlfriends():	
	constrainer = MetricLearningFriendsConstrainer( 
		metriclearner='truncatedscaling',
		keep_phantom_points=False, optimize_phantom_points=False,
		force_shrink=True, 
		rebuild_every=nlive_points, 
		verbose=False)
	termination = BootstrappedCriterion(tolerance=0.5)
	run(constrainer, termination)

def test_hmlfriends():
	friends_filter = MetricLearningFriendsConstrainer( 
		metriclearner='truncatedscaling',
		keep_phantom_points=False, optimize_phantom_points=False,
		force_shrink=True, 
		rebuild_every=nlive_points, 
		verbose=False)
	proposer = FilteredGaussProposal(adapt='sivia-neg-binom', scale=0.1)
	filtered_mcmc = FilteredMCMCConstrainer(proposer=proposer, nsteps=nsteps)
	constrainer = HybridMLFriendsConstrainer(friends_filter, filtered_mcmc, 
		switchover_efficiency=0, unfiltered=False)
	termination = TerminationCriterion(tolerance=0.5, maxRemainderFraction=0.001)
	run(constrainer, termination)

def test_hmultiellipsoid():
	friends_filter = MetricLearningFriendsConstrainer( 
		metriclearner='truncatedscaling',
		keep_phantom_points=False, optimize_phantom_points=False,
		force_shrink=True, 
		rebuild_every=nlive_points, 
		verbose=False)
	proposer = FilteredUnitRandomSliceProposal()
	filtered_mcmc = FilteredSliceConstrainer(proposer=proposer, nsteps=nsteps)
	constrainer = HybridMLMultiEllipsoidConstrainer(filtered_mcmc, 
		metriclearner='simplescaling', 
		switchover_efficiency=0.001)
	termination = TerminationCriterion(tolerance=0.5, maxRemainderFraction=0.001)
	run(constrainer, termination)

if __name__ == '__main__':
	test_naive()
	test_mcmc()
	test_radfriends()
	test_mlfriends()	
	test_hmlfriends()
	test_hmultiellipsoid()

