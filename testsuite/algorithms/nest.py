"""
Runs Python-implemented Nested Sampling algorithms from this package.
"""
import json
import itertools
import numpy
from numpy import log, exp

from nested_sampling.nested_integrator import nested_integrator
from nested_sampling.nested_sampler import NestedSampler
from nested_sampling.samplers.rejection import RejectionConstrainer
#from nested_sampling.samplers.friendsnext import FriendsConstrainer as FriendsConstrainer2
from nested_sampling.samplers.friends import FriendsConstrainer
FriendsConstrainer2 = FriendsConstrainer
from nested_sampling.samplers.hiermetriclearn import MetricLearningFriendsConstrainer
from nested_sampling.samplers.optimize import OptimizeConstrainer
from nested_sampling.samplers.mcmc import MCMCConstrainer, GaussProposal, MultiScaleProposal
#from nested_sampling.samplers.galilean import GalileanConstrainer
from nested_sampling.samplers.svm.svmnest import SVMConstrainer
from nested_sampling.samplers.ellipsoidal import EllipsoidConstrainer, MultiEllipsoidConstrainer
import nested_sampling.samplers.hybrid
import matplotlib.pyplot as plt
from nested_sampling.postprocess import equal_weighted_posterior, marginal_plots
import time
from nested_sampling.termination_criteria import TerminationCriterion, MaxErrorCriterion, BootstrappedCriterion, RememberingBootstrappedCriterion, DecliningBootstrappedCriterion, NoisyBootstrappedCriterion, NoiseDetectingBootstrappedCriterion

def run_nested(**config):
	ndim = config['ndim']
	
	def priortransform(u):
		assert len(u) == ndim, u
		return u
	if 'seed' in config:
		numpy.random.seed(config['seed'])

	print 'Configuring for %s, with seed=%s ...' % (config.get('output_basename'), config.get('seed'))
	# can use these directly
	loglikelihood = config['loglikelihood']
	nlive_points = config['nlive_points']
	method = config['draw_method']
	if method.startswith('naive'):
		constrainer = RejectionConstrainer()
	elif method.startswith('maxfriends'): # maximum distance
		constrainer = FriendsConstrainer(rebuild_every=nlive_points, radial=False, force_shrink=config['force_shrink'], verbose=False)
	elif method.startswith('radfriends2'): # radial distance
		constrainer = FriendsConstrainer2(rebuild_every=nlive_points, radial=True, metric = 'euclidean', jackknife=config['jackknife'], force_shrink=config['force_shrink'], verbose=False)
	elif method.startswith('supfriends2'): # supreme distance
		constrainer = FriendsConstrainer2(rebuild_every=nlive_points, radial=True, metric = 'chebyshev', jackknife=config['jackknife'], force_shrink=config['force_shrink'], verbose=False)
	elif method.startswith('radfriends'): # radial distance
		constrainer = FriendsConstrainer(rebuild_every=nlive_points, radial=True, metric = 'euclidean', jackknife=config['jackknife'], force_shrink=config['force_shrink'], keep_phantom_points=config.get('keep_phantom_points', False), optimize_phantom_points=config.get('optimize_phantom_points', False), verbose=False)
	elif method.startswith('mlfriends'): # metric-learning distance
		constrainer = MetricLearningFriendsConstrainer( 
			metriclearner=config['metriclearner'],
			keep_phantom_points=config.get('keep_phantom_points', False), 
			optimize_phantom_points=config.get('optimize_phantom_points', False), 
			force_shrink=config['force_shrink'], 
			rebuild_every=config.get('rebuild_every', nlive_points), 
			verbose=False)
	elif method.startswith('hradfriends'): # radial distance
		friends_filter = FriendsConstrainer(rebuild_every=nlive_points, radial=True, metric = 'euclidean', jackknife=config['jackknife'], force_shrink=config['force_shrink'], keep_phantom_points=config.get('keep_phantom_points', False), optimize_phantom_points=config.get('optimize_phantom_points', False), verbose=False)
		if config['proposer'] == 'gauss':
			proposer = nested_sampling.samplers.hybrid.FilteredGaussProposal(adapt=True, scale=0.1)
		elif config['proposer'] == 'svargauss':
			proposer = nested_sampling.samplers.hybrid.FilteredSVarGaussProposal(adapt=True, scale=0.1)
		elif config['proposer'] == 'mahgauss':
			proposer = nested_sampling.samplers.hybrid.FilteredMahalanobisGaussProposal(adapt=True, scale=0.1)
		elif config['proposer'] == 'harm':
			proposer = nested_sampling.samplers.hybrid.FilteredUnitHARMProposal(adapt=False, scale=1)
		elif config['proposer'] == 'mahharm':
			proposer = nested_sampling.samplers.hybrid.FilteredMahalanobisHARMProposal(adapt=False, scale=1)
		elif config['proposer'] == 'ptharm':
			proposer = nested_sampling.samplers.hybrid.FilteredPointHARMProposal(adapt=False, scale=10)
		elif config['proposer'] == 'ess':
			proposer = nested_sampling.samplers.hybrid.FilteredEllipticalSliceProposal()
		else:
			assert False, config['proposer']
		if config['nsteps'] < 0:
			filtered_mcmc = nested_sampling.samplers.hybrid.FilteredVarlengthMCMCConstrainer(proposer=proposer, 
				nsteps_initial=-config['nsteps'])
		else:
			filtered_mcmc = nested_sampling.samplers.hybrid.FilteredMCMCConstrainer(proposer=proposer, 
				nsteps=config['nsteps'], nminaccepts=config.get('nminaccepts', 0))
		constrainer = nested_sampling.samplers.hybrid.HybridFriendsConstrainer(friends_filter, filtered_mcmc, 
			switchover_efficiency=config.get('switchover_efficiency', 0))
	elif method.startswith('hmlfriends'): # radial distance
		friends_filter = MetricLearningFriendsConstrainer(rebuild_every=nlive_points, 
			metriclearner=config['metriclearner'],
			keep_phantom_points=config.get('keep_phantom_points', False), 
			optimize_phantom_points=config.get('optimize_phantom_points', False), 
			force_shrink=config['force_shrink'], 
			verbose=False)
		if config['proposer'] == 'gauss':
			proposer = nested_sampling.samplers.hybrid.FilteredGaussProposal(adapt=True, scale=0.1)
		elif config['proposer'] == 'harm':
			proposer = nested_sampling.samplers.hybrid.FilteredUnitHARMProposal(adapt=False, scale=1)
		elif config['proposer'] == 'mahharm':
			proposer = nested_sampling.samplers.hybrid.FilteredMahalanobisHARMProposal(adapt=False, scale=1)
		elif config['proposer'] == 'ptharm':
			proposer = nested_sampling.samplers.hybrid.FilteredPointHARMProposal(adapt=False, scale=10)
		elif config['proposer'] == 'diffptharm':
			proposer = nested_sampling.samplers.hybrid.FilteredDeltaPointHARMProposal(adapt=False, scale=10)
		elif config['proposer'] == 'ess':
			proposer = nested_sampling.samplers.hybrid.FilteredEllipticalSliceProposal()
		else:
			assert False, config['proposer']
		if config['nsteps'] < 0:
			filtered_mcmc = nested_sampling.samplers.hybrid.FilteredVarlengthMCMCConstrainer(proposer=proposer, 
				nsteps_initial=-config['nsteps'])
		else:
			filtered_mcmc = nested_sampling.samplers.hybrid.FilteredMCMCConstrainer(proposer=proposer, 
				nsteps=config['nsteps'], nminaccepts=config.get('nminaccepts', 0))
		constrainer = nested_sampling.samplers.hybrid.HybridMLFriendsConstrainer(friends_filter, filtered_mcmc, 
			switchover_efficiency=config.get('switchover_efficiency', 0), 
			unfiltered=config.get('unfiltered', False))
	elif method.startswith('hmultiellipsoid'): # multi-ellipsoid
		if config['proposer'] == 'gauss':
			proposer = nested_sampling.samplers.hybrid.FilteredGaussProposal(adapt=True, scale=0.1)
		elif config['proposer'] == 'svargauss':
			proposer = nested_sampling.samplers.hybrid.FilteredSVarGaussProposal(adapt=True, scale=0.1)
		elif config['proposer'] == 'mahgauss':
			proposer = nested_sampling.samplers.hybrid.FilteredMahalanobisGaussProposal(adapt=True, scale=0.1)
		elif config['proposer'] == 'harm':
			proposer = nested_sampling.samplers.hybrid.FilteredUnitHARMProposal(adapt=False, scale=1)
		elif config['proposer'] == 'mahharm':
			proposer = nested_sampling.samplers.hybrid.FilteredMahalanobisHARMProposal(adapt=False, scale=1)
		elif config['proposer'] == 'ptharm':
			proposer = nested_sampling.samplers.hybrid.FilteredPointHARMProposal(adapt=False, scale=10)
		elif config['proposer'] == 'diffptharm':
			proposer = nested_sampling.samplers.hybrid.FilteredDeltaPointHARMProposal(adapt=False, scale=10)
		elif config['proposer'] == 'ess':
			proposer = nested_sampling.samplers.hybrid.FilteredEllipticalSliceProposal()
		else:
			assert False, config['proposer']
		if config['nsteps'] < 0:
			filtered_mcmc = nested_sampling.samplers.hybrid.FilteredVarlengthMCMCConstrainer(proposer=proposer, 
				nsteps_initial=-config['nsteps'])
		else:
			filtered_mcmc = nested_sampling.samplers.hybrid.FilteredMCMCConstrainer(proposer=proposer, 
				nsteps=config['nsteps'], nminaccepts=config.get('nminaccepts', 0))
		constrainer = nested_sampling.samplers.hybrid.HybridMultiEllipsoidConstrainer(filtered_mcmc, enlarge=config.get('enlarge', 1.2), 
			switchover_efficiency=config.get('switchover_efficiency', 0))
	elif method.startswith('hmlmultiellipsoid'): # multi-ellipsoid
		if config['proposer'] == 'gauss':
			proposer = nested_sampling.samplers.hybrid.FilteredGaussProposal(adapt=True, scale=0.1)
		elif config['proposer'] == 'svargauss':
			proposer = nested_sampling.samplers.hybrid.FilteredSVarGaussProposal(adapt=True, scale=0.1)
		elif config['proposer'] == 'mahgauss':
			proposer = nested_sampling.samplers.hybrid.FilteredMahalanobisGaussProposal(adapt=True, scale=0.1)
		elif config['proposer'] == 'harm':
			proposer = nested_sampling.samplers.hybrid.FilteredUnitHARMProposal(adapt=False, scale=1)
		elif config['proposer'] == 'mahharm':
			proposer = nested_sampling.samplers.hybrid.FilteredMahalanobisHARMProposal(adapt=False, scale=1)
		elif config['proposer'] == 'ptharm':
			proposer = nested_sampling.samplers.hybrid.FilteredPointHARMProposal(adapt=False, scale=10)
		elif config['proposer'] == 'diffptharm':
			proposer = nested_sampling.samplers.hybrid.FilteredDeltaPointHARMProposal(adapt=False, scale=10)
		elif config['proposer'] == 'ess':
			proposer = nested_sampling.samplers.hybrid.FilteredEllipticalSliceProposal()
		else:
			assert False, config['proposer']
		if config['nsteps'] < 0:
			filtered_mcmc = nested_sampling.samplers.hybrid.FilteredVarlengthMCMCConstrainer(proposer=proposer, 
				nsteps_initial=-config['nsteps'])
		else:
			filtered_mcmc = nested_sampling.samplers.hybrid.FilteredMCMCConstrainer(proposer=proposer, 
				nsteps=config['nsteps'], nminaccepts=config.get('nminaccepts', 0))
		constrainer = nested_sampling.samplers.hybrid.HybridMLMultiEllipsoidConstrainer(filtered_mcmc, 
			metriclearner=config['metriclearner'], 
			switchover_efficiency=config.get('switchover_efficiency', 0),
			enlarge=config.get('enlarge', 1.2),
			bs_enabled=config.get('bs_enabled', False),
			)
	elif method.startswith('supfriends'): # supreme distance
		constrainer = FriendsConstrainer(rebuild_every=nlive_points, radial=True, metric = 'chebyshev', jackknife=config['jackknife'], force_shrink=config['force_shrink'], verbose=False)
	# These two do not work
	# Because after an update, at a later time, the distances computed are rescaled based on new points
	# we would need to store the metric at update time
	#elif method.startswith('sradfriends'): 
	#	constrainer = FriendsConstrainer(rebuild_every=nlive_points, radial=True, metric = 'seuclidean', jackknife=config['jackknife'], force_shrink=config['force_shrink'], verbose=False)
	#elif method.startswith('mahfriends'): 
	#	constrainer = FriendsConstrainer(rebuild_every=nlive_points, radial=True, metric = 'mahalanobis', jackknife=config['jackknife'], force_shrink=config['force_shrink'], verbose=False)
	elif method.startswith('optimize'):
		constrainer = OptimizeConstrainer()
	elif method.startswith('ellipsoid'):
		constrainer = EllipsoidConstrainer()
	elif method.startswith('multiellipsoid'):
		constrainer = MultiEllipsoidConstrainer()
	elif method.startswith('galilean'):
		velocity_scale = config['velocity_scale']
		constrainer = GalileanConstrainer(nlive_points = nlive_points, ndim = ndim, velocity_scale = velocity_scale)
	elif method.startswith('mcmc'):
		adapt = config['adapt']
		scale = config['scale']
		if config['proposer'] == 'gauss':
			proposer = GaussProposal(adapt=adapt, scale = scale)
		elif config['proposer'] == 'multiscale':
			proposer = MultiScaleProposal(adapt=adapt, scale=scale)
		constrainer = MCMCConstrainer(proposer = proposer, nsteps=config['nsteps'], nminaccepts=config.get('nminaccepts', 0))
	else:
		raise NotImplementedError('draw_method "%s" not implemented' % method)

	print 'configuring TerminationCriterion'
	if config.get('unlimited_sampling', False):
		max_samples = None
	else:
		max_samples = 2000000
	
	if config['integrator'] == 'normal':
		termination = TerminationCriterion(tolerance=0.5)
	elif config['integrator'] == 'normal-max':
		termination = MaxErrorCriterion(tolerance=0.5)
	elif config['integrator'] == 'normal-verysmall':
		termination = TerminationCriterion(tolerance=0.5, maxRemainderFraction=0.001)
	elif config['integrator'] == 'normal-bs':
		termination = BootstrappedCriterion(tolerance=0.5)
		#result = nested_integrator(tolerance=0.5, sampler=sampler, max_samples=max_samples, need_small_remainder=False, need_robust_remainder_error=True)
	elif config['integrator'] == 'normal+bs2':
		termination = BootstrappedCriterion(tolerance=0.5, maxRemainderFraction=0.5)
	elif config['integrator'] == 'normal+bs3':
		termination = BootstrappedCriterion(tolerance=0.5, maxRemainderFraction=1/3.)
	elif config['integrator'] == 'normal+bs10':
		termination = BootstrappedCriterion(tolerance=0.5, maxRemainderFraction=1/10.)
	elif config['integrator'] == 'normal-rbs3':
		termination = RememberingBootstrappedCriterion(tolerance=0.5, memory_length=3)
	elif config['integrator'] == 'normal-rbs5':
		termination = RememberingBootstrappedCriterion(tolerance=0.5, memory_length=5)
	elif config['integrator'] == 'normal+rbs32':
		termination = RememberingBootstrappedCriterion(tolerance=0.5, memory_length=3, maxRemainderFraction=0.5)
	elif config['integrator'] == 'normal-dbs11':
		termination = DecliningBootstrappedCriterion(tolerance=0.5, required_decrease=1., required_decrease_scatter=1.)
	elif config['integrator'] == 'normal-dbs22':
		termination = DecliningBootstrappedCriterion(tolerance=0.5, required_decrease=0.5, required_decrease_scatter=0.5)
	#elif config['integrator'] == 'normal-dbs31':
	#	termination = DecliningBootstrappedCriterion(tolerance=0.5, required_decrease=1./3., required_decrease_scatter=1.)
	elif config['integrator'] == 'normal-dbs33':
		termination = DecliningBootstrappedCriterion(tolerance=0.5, required_decrease=1./3., required_decrease_scatter=1./3.)
	elif config['integrator'] == 'normal-dbs03':
		termination = DecliningBootstrappedCriterion(tolerance=0.5, required_decrease=0., required_decrease_scatter=1./3.)
	elif config['integrator'] == 'normal-dbs01':
		termination = DecliningBootstrappedCriterion(tolerance=0.5, required_decrease=0., required_decrease_scatter=1.)
	elif config['integrator'] == 'normal-dbs10':
		termination = DecliningBootstrappedCriterion(tolerance=0.5, required_decrease=1., required_decrease_scatter=0.)
	elif config['integrator'] == 'normal-nbs':
		termination = NoisyBootstrappedCriterion(tolerance=0.5)
	elif config['integrator'] == 'normal-cnbs':
		termination = NoisyBootstrappedCriterion(tolerance=0.5, conservative=True)
	elif config['integrator'] == 'normal-ndbs10':
		termination = NoiseDetectingBootstrappedCriterion(tolerance=0.5, maxNoisyRemainder=0.1)
	elif config['integrator'] == 'normal-ndbs100':
		termination = NoiseDetectingBootstrappedCriterion(tolerance=0.5, maxNoisyRemainder=0.01)
	else:
		assert config['integrator'] == 'normal', config['integrator']
	# only record for the first seed
	termination.plot = config.get('seed', 0) == 0
	
	print 'configuring NestedSampler'
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
	print 'running nested_integrator to tolerance 0.5'
	result = nested_integrator(sampler=sampler, max_samples=max_samples, terminationcriterion=termination)

	endtime = time.time()
	if hasattr(constrainer, 'stats'):
		constrainer.stats()

	output_basename = config['output_basename']
	#numpy.savetxt(output_basename + 'convergencetests.txt.gz', result['convergence_tests'])
	
	if config.get('seed', 0) == 0:
		# drawn samples
		print 'plotting drawn samples...'
		x = numpy.array([x for _, x, _ in sampler.samples])
		y = exp([l for _, _, l in sampler.samples])
		plt.plot(x[:,0], y, 'x', color='blue', ms=1)
		plt.savefig(output_basename + 'nested_samples.pdf', bbox_inches='tight')
		plt.close()
		
		# L vs V
		print 'plotting V-L...'
		L = numpy.array([L for _, _, L, _ in result['weights']])
		width = numpy.array([w for _, _, _, w in result['weights']])
		plt.plot(width, L, 'x-', color='blue', ms=1, label='Z=%.2f (%.2f)' % (
			result['logZ'], log(exp(L + width).sum())))
		fromleft = exp(L + width)[::-1].cumsum()
		fromleft /= fromleft.max()
		mask = (fromleft < 0.99)[::-1]
		if mask.any():
			i = width[mask].argmax()
			plt.ylim(L.max() - log(1000), L.max())
			plt.fill_between(width[mask], L[mask], L.max() - log(1000), color='grey', alpha=0.3)
		plt.xlabel('prior mass')
		plt.ylabel('likelihood')
		plt.legend(loc='best')
		plt.savefig(output_basename + 'nested_integral.pdf', bbox_inches='tight')
		plt.close()
		
		# posteriors
		print 'plotting posteriors...'
		posterioru, posteriorx = equal_weighted_posterior(result['weights'])
		plt.figure(figsize=(ndim*2, ndim*2))
		marginal_plots(weights=result['weights'], ndim=ndim)
		plt.savefig(output_basename + 'posterior.pdf', bbox_inches='tight')
		plt.close()
		
		# plot convergence history
		print 'plotting Z history...'
		plt.figure()
		plt.plot(termination.plotdata['normalZ'], label='NS')
		plt.plot(termination.plotdata['remainderZ'], label='remainder')
		plt.plot(termination.plotdata['totalZ'], label='total')
		hi = max(termination.plotdata['totalZ'])
		plt.ylim(hi - 10, hi + 0.1)
		plt.legend(loc='best', prop=dict(size=8))
		plt.savefig(output_basename + 'convergence_Z.pdf', bbox_inches='tight')
		plt.close()

		print 'plotting convergence history...'
		plt.figure()
		plt.plot(termination.plotdata['normalZerr'], label='NS')
		plt.plot(termination.plotdata['remainderZerr'], label='remainder')
		plt.plot(termination.plotdata['totalZerr'], label='total')
		if 'memory_sigma' in termination.plotdata:
			plt.plot(termination.plotdata['memory_sigma'], label='memory_sigma')
		if 'classic_totalZerr' in termination.plotdata:
			plt.plot(termination.plotdata['classic_totalZerr'], label='classic_totalZerr')
		plt.ylim(0, 2)
		plt.legend(loc='best', prop=dict(size=8))
		plt.savefig(output_basename + 'convergence_Zerr.pdf', bbox_inches='tight')
		plt.close()

	return dict(
		Z_computed = float(result['logZ']),
		Z_computed_err = float(result['logZerr']),
		niterations = result['niterations'],
		duration = endtime - starttime,
	)

configs = [
	[
		#dict(nlive_points=100),
		dict(nlive_points=400),
		dict(nlive_points=1000),
	], [
		#dict(draw_method='naive'),
		#dict(draw_method='optimizer', constrainer=OptimizeConstrainer()),
		#dict(draw_method='maxfriends'),
		#dict(draw_method='radfriends', jackknife=False, force_shrink=False),
		#dict(draw_method='supfriends', jackknife=False, force_shrink=False),
		#dict(draw_method='mlfriends', metriclearner='simplescaling', force_shrink=False),
		#dict(draw_method='mlfriends-phantoms', metriclearner='simplescaling', keep_phantom_points=True, force_shrink=True),
		#dict(draw_method='mlfriends-optphantoms', metriclearner='simplescaling', keep_phantom_points=True, optimize_phantom_points=True, force_shrink=True),
		#dict(draw_method='mlfriendsT', metriclearner='truncatedscaling', force_shrink=True),
		dict(draw_method='mlfriendsTM', metriclearner='truncatedmahalanobis', force_shrink=True),
		#dict(draw_method='mlfriendsTMR', metriclearner='truncatedmahalanobis', force_shrink=False, rebuild_every=10),
		#dict(draw_method='mlfriendsTSDML', metriclearner='truncatedsdml', force_shrink=True),
		#dict(draw_method='mlfriendsT-phantoms', metriclearner='truncatedscaling', keep_phantom_points=True, force_shrink=True),
		#dict(draw_method='mlfriendsT-optphantoms', metriclearner='truncatedscaling', keep_phantom_points=True, optimize_phantom_points=True, force_shrink=True),
		#dict(draw_method='mlfriends-phantoms', metriclearner='simplescaling', keep_phantom_points=True),
		#dict(draw_method='sradfriends', jackknife=False, force_shrink=False),
		#dict(draw_method='mahfriends', jackknife=False, force_shrink=False),
		#dict(draw_method='radfriends2', jackknife=False, force_shrink=False),
		#dict(draw_method='supfriends2', jackknife=False, force_shrink=False),
		#dict(draw_method='radfriends-phantoms', jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='radfriends-optphantoms', jackknife=False, force_shrink=True, keep_phantom_points=True, optimize_phantom_points=True),
		#dict(draw_method='radfriends-1', jackknife=True, force_shrink=True),
		#dict(draw_method='radfriends-1-phantoms', jackknife=True, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='radfriends-1-optphantoms', jackknife=True, force_shrink=True, keep_phantom_points=True, optimize_phantom_points=True),

		#dict(draw_method='mcmc-gauss-scale0.1', proposer = 'gauss', adapt=False, scale=0.1),
		#dict(draw_method='mcmc-gauss-adapt-5steps', proposer = 'gauss', nsteps=5, adapt=True, scale=0.1),
		#dict(draw_method='mcmc-gauss-adapt-10steps', proposer = 'gauss', nsteps=10, adapt=True, scale=0.1),
		#dict(draw_method='mcmc-gauss-adapt-20steps', proposer = 'gauss', nsteps=20, adapt=True, scale=0.1),
		#dict(draw_method='mcmc-gauss-adapt+5steps',  proposer = 'gauss', nsteps=5, nminaccepts=5, adapt=True, scale=0.1),
		#dict(draw_method='mcmc-gauss-adapt+20steps', proposer = 'gauss', nsteps=20, nminaccepts=20, adapt=True, scale=0.1),
		#dict(draw_method='hradfriends-gauss-5steps-phantoms', proposer = 'gauss', nsteps=5, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-gauss-10steps-phantoms', proposer = 'gauss', nsteps=10, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-svargauss-5steps-phantoms', proposer = 'svargauss', nsteps=5, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-svargauss-10steps-phantoms', proposer = 'svargauss', nsteps=10, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-mahgauss-5steps-phantoms', proposer = 'mahgauss', nsteps=5, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-mahgauss-10steps-phantoms', proposer = 'mahgauss', nsteps=10, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-gauss+2steps-phantoms', proposer = 'gauss', nsteps=2, nminaccepts=2, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-gauss+5steps-phantoms', proposer = 'gauss', nsteps=5, nminaccepts=5, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-svargauss+2steps-phantoms', proposer = 'svargauss', nsteps=2, nminaccepts=2, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-svargauss+5steps-phantoms', proposer = 'svargauss', nsteps=5, nminaccepts=5, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-mahgauss+2steps-phantoms', proposer = 'mahgauss', nsteps=2, nminaccepts=2, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-mahgauss+5steps-phantoms', proposer = 'mahgauss', nsteps=5, nminaccepts=5, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-harm-5steps-phantoms', proposer = 'harm', nsteps=5, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-harm+2steps-phantoms', proposer = 'harm', nsteps=2, nminaccepts=2, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-harm+5steps-phantoms', proposer = 'harm', nsteps=5, nminaccepts=5, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-harm-10steps-phantoms', proposer = 'harm', nsteps=10, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-harm+10steps-phantoms', proposer = 'harm', nsteps=10, nminaccepts=10, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-harm-50steps-phantoms', proposer = 'harm', nsteps=50, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-mahharm+5steps-phantoms', proposer = 'mahharm', nsteps=5, nminaccepts=5, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-mahharm+10steps-phantoms', proposer = 'mahharm', nsteps=10, nminaccepts=10, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-ptharm+5steps-phantoms', proposer = 'ptharm', nsteps=5, nminaccepts=5, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-ptharm+10steps-phantoms', proposer = 'ptharm', nsteps=10, nminaccepts=10, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-ess+5steps-phantoms', proposer = 'ess', nsteps=5, nminaccepts=5, jackknife=False, force_shrink=True, keep_phantom_points=True),
		#dict(draw_method='hradfriends-ess+10steps-phantoms', proposer = 'ess', nsteps=10, nminaccepts=10, jackknife=False, force_shrink=True, keep_phantom_points=True),
		
		#dict(draw_method='hmultiellipsoid-gauss+5steps', proposer = 'gauss', nsteps=10, nminaccepts=10),
		#dict(draw_method='hmultiellipsoid-mahgauss+5steps', proposer = 'mahgauss', nsteps=5, nminaccepts=5),
		#dict(draw_method='hmultiellipsoid2-harm+5steps', proposer = 'harm', nsteps=5, nminaccepts=5),
		#dict(draw_method='hmultiellipsoid2-harm+2steps', proposer = 'harm', nsteps=2, nminaccepts=2),
		#dict(draw_method='hmultiellipsoid2-harm+1steps', proposer = 'harm', nsteps=1, nminaccepts=1),
		#dict(draw_method='hmlmultiellipsoid-harm+2steps', metriclearner='simplescaling', proposer = 'harm', nsteps=2, nminaccepts=2),
		#dict(draw_method='hmlmultiellipsoid-harm+1steps', metriclearner='simplescaling', proposer = 'harm', nsteps=1, nminaccepts=1),
		#dict(draw_method='hmultiellipsoidBS-harm+5steps', proposer = 'harm', nsteps=5, nminaccepts=5),
		#dict(draw_method='hmultiellipsoidBS-harm+2steps', proposer = 'harm', nsteps=2, nminaccepts=2),
		#dict(draw_method='hmultiellipsoidBS-harm+1steps', proposer = 'harm', nsteps=1, nminaccepts=1),
		#dict(draw_method='hmlmultiellipsoidBS-harm+2steps', metriclearner='simplescaling', proposer = 'harm', nsteps=2, nminaccepts=2, bs_enabled=True),
		#dict(draw_method='hmlmultiellipsoidBS-harm+1steps', metriclearner='simplescaling', proposer = 'harm', nsteps=1, nminaccepts=1, bs_enabled=True),
		dict(draw_method='hmlmultiellipsoidBSM-harm+1steps', metriclearner='mahalanobis', proposer = 'harm', nsteps=1, nminaccepts=1, bs_enabled=True),
		dict(draw_method='hmlmultiellipsoidBSM-harm+2steps', metriclearner='mahalanobis', proposer = 'harm', nsteps=2, nminaccepts=2, bs_enabled=True),
		#dict(draw_method='hmlmultiellipsoidBSSDML-harm+1steps', metriclearner='sdml', proposer = 'harm', nsteps=1, nminaccepts=1, bs_enabled=True),
		#dict(draw_method='hmlmultiellipsoidBSSDML-harm+2steps', metriclearner='sdml', proposer = 'harm', nsteps=2, nminaccepts=2, bs_enabled=True),
		#dict(draw_method='hmlmultiellipsoidBSM-harm+varsteps', metriclearner='mahalanobis', proposer = 'harm', nsteps=-2, bs_enabled=True),
		#dict(draw_method='hmlmultiellipsoidBSM-harm+5steps', metriclearner='mahalanobis', proposer = 'harm', nsteps=5, nminaccepts=5, bs_enabled=True),
		#dict(draw_method='hmlmultiellipsoidBSM-ess+1steps', metriclearner='mahalanobis', proposer = 'ess', nsteps=1, nminaccepts=1, bs_enabled=True),
		#dict(draw_method='hmlmultiellipsoidBSM-ess+2steps', metriclearner='mahalanobis', proposer = 'ess', nsteps=2, nminaccepts=2, bs_enabled=True),
		#dict(draw_method='hmlmultiellipsoidBSM-ess+5steps', metriclearner='mahalanobis', proposer = 'ess', nsteps=5, nminaccepts=5, bs_enabled=True),
		#dict(draw_method='hmultiellipsoid-diffptharm+2steps', proposer = 'diffptharm', nsteps=2, nminaccepts=2),
		#dict(draw_method='hmultiellipsoid-diffptharm+1steps', proposer = 'diffptharm', nsteps=1, nminaccepts=1),
		#dict(draw_method='hmultiellipsoid-harm+10steps', proposer = 'harm', nsteps=10, nminaccepts=10),
		#dict(draw_method='hmultiellipsoid-mahharm+5steps', proposer = 'mahharm', nsteps=5, nminaccepts=5),
		#dict(draw_method='hmultiellipsoid-mahharm+2steps', proposer = 'mahharm', nsteps=2, nminaccepts=2),
		#dict(draw_method='hmultiellipsoid-ptharm+5steps', proposer = 'ptharm', nsteps=5, nminaccepts=5),
		#dict(draw_method='hmultiellipsoid-ess+5steps',    proposer = 'ess',    nsteps=5, nminaccepts=5),

		#dict(draw_method='hmlfriends-gauss+5steps-phantoms', metriclearner='truncatedscaling', proposer = 'gauss', nsteps=5, nminaccepts=5, keep_phantom_points=True),
		#dict(draw_method='hmlfriends-harm+5steps-phantoms', metriclearner='simplescaling', proposer = 'harm', nsteps=5, nminaccepts=5, keep_phantom_points=True, force_shrink=False),
		#dict(draw_method='hmlfriends-ptharm+5steps-phantoms', metriclearner='truncatedscaling', proposer = 'ptharm', nsteps=5, nminaccepts=5, keep_phantom_points=True),
		#dict(draw_method='hmlfriendsT-gauss+5steps-phantoms', metriclearner='truncatedscaling', proposer = 'gauss', nsteps=5, nminaccepts=5, keep_phantom_points=True, force_shrink=True),
		#dict(draw_method='hmlfriendsTM-harm+5steps-optphantoms', metriclearner='truncatedmahalanobis', proposer = 'harm', nsteps=5, nminaccepts=5, keep_phantom_points=True, optimize_phantom_points=True, force_shrink=True),
		#dict(draw_method='hmlfriendsT-harm+2steps-optphantoms', metriclearner='truncatedscaling', proposer = 'harm', nsteps=2, nminaccepts=2, keep_phantom_points=True, optimize_phantom_points=True, force_shrink=True),
		#dict(draw_method='hmlfriendsT-harm+1steps-optphantoms', metriclearner='truncatedscaling', proposer = 'harm', nsteps=1, nminaccepts=1, keep_phantom_points=True, optimize_phantom_points=True, force_shrink=True),
		#dict(draw_method='hmlfriendsT-switch400-harm+5steps-phantoms', metriclearner='truncatedscaling', proposer = 'harm', nsteps=5, nminaccepts=5, keep_phantom_points=True, force_shrink=True, switchover_efficiency=1./400),
		#dict(draw_method='hmlfriendsT-switch40-harm+5steps-phantoms', metriclearner='truncatedscaling', proposer = 'harm', nsteps=5, nminaccepts=5, keep_phantom_points=True, force_shrink=True, switchover_efficiency=1./40),
		#dict(draw_method='hmlfriendsT-ptharm+5steps-phantoms', metriclearner='truncatedscaling', proposer = 'ptharm', nsteps=5, nminaccepts=5, keep_phantom_points=True, force_shrink=True),
		dict(draw_method='hmlfriendsTM-harm+1steps', metriclearner='truncatedmahalanobis', proposer = 'harm', nsteps=1, nminaccepts=1, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),
		dict(draw_method='hmlfriendsTM-harm+2steps', metriclearner='truncatedmahalanobis', proposer = 'harm', nsteps=2, nminaccepts=2, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),
		#dict(draw_method='hmlfriendsTSDML-harm+1steps', metriclearner='truncatedsdml', proposer = 'harm', nsteps=1, nminaccepts=1, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),
		#dict(draw_method='hmlfriendsTSDML-harm+2steps', metriclearner='truncatedsdml', proposer = 'harm', nsteps=2, nminaccepts=2, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),
		#dict(draw_method='hmlfriendsTM-harm+varsteps', metriclearner='truncatedmahalanobis', proposer = 'harm', nsteps=-2, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),
		#dict(draw_method='hmlfriendsTM-harm+5steps', metriclearner='truncatedmahalanobis', proposer = 'harm', nsteps=5, nminaccepts=5, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),
		#dict(draw_method='hmlfriendsTM-ptharm+1steps', metriclearner='truncatedmahalanobis', proposer = 'ptharm', nsteps=1, nminaccepts=1, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),
		#dict(draw_method='hmlfriendsTM-mahharm+1steps', metriclearner='truncatedmahalanobis', proposer = 'mahharm', nsteps=1, nminaccepts=1, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),
		#dict(draw_method='hmlfriendsTM-mahharm+2steps', metriclearner='truncatedmahalanobis', proposer = 'mahharm', nsteps=2, nminaccepts=2, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),
		#dict(draw_method='hmlfriendsTM-ptharm+5steps', metriclearner='truncatedmahalanobis', proposer = 'ptharm', nsteps=5, nminaccepts=5, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),
		#dict(draw_method='hmlfriendsTM-ess+1steps', metriclearner='truncatedmahalanobis', proposer = 'ess', nsteps=1, nminaccepts=1, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),
		#dict(draw_method='hmlfriendsTM-ess+2steps', metriclearner='truncatedmahalanobis', proposer = 'ess', nsteps=2, nminaccepts=2, keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True),

		#dict(draw_method='mcmc-multiscale-scale3', proposer = 'multiscale', adapt=False, scale=3),
		#dict(draw_method='mcmc-multiscale-scale3-adapt', proposer = 'multiscale', adapt=True, scale=3),
		#dict(draw_method='galilean-velocity2', velocity_scale = 0.03),
		#dict(draw_method='galilean-velocity1', velocity_scale = 0.1),
		#dict(draw_method='galilean-velocity3', velocity_scale = 0.001),
		#dict(draw_method='galilean-velocity0', velocity_scale = 0.3),
		#dict(draw_method='hiercluster-svm', constrainer=SVMConstrainer()),
		dict(draw_method='ellipsoidal'),
		dict(draw_method='multiellipsoidal'),
	], [
		dict(integrator='normal'), 
		dict(integrator='normal-bs'), 
		#dict(integrator='normal+bs2'), 
		#dict(integrator='normal+bs3'), 
		dict(integrator='normal+bs10'), 
		#dict(integrator='normal-rbs3'), 
		#dict(integrator='normal-rbs5'), 
		#dict(integrator='normal+rbs32'),
		dict(integrator='normal-dbs11'),
		dict(integrator='normal-dbs33'),
		dict(integrator='normal-dbs03'),
		dict(integrator='normal-dbs01'),
		dict(integrator='normal-dbs10'),
		dict(integrator='normal-nbs'),
		dict(integrator='normal-cnbs'),
		dict(integrator='normal-ndbs10'),
		dict(integrator='normal-ndbs100'),
		dict(integrator='normal-max'), 
	]
]
configs = [dict([[k, v] for d in config for k, v in d.iteritems()]) for config in itertools.product(*configs)]
for c in configs:
	c['algorithm_name'] = 'NS-%s-nlive%d%s' % (c['draw_method'], c['nlive_points'],
		('-' + c['integrator']) if c['integrator'] != 'normal' else '')
	c['run'] = run_nested



