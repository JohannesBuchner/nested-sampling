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
from nested_sampling.samplers.optimize import OptimizeConstrainer
from nested_sampling.samplers.mcmc import MCMCConstrainer, GaussProposal, MultiScaleProposal
#from nested_sampling.samplers.galilean import GalileanConstrainer
from nested_sampling.samplers.svm.svmnest import SVMConstrainer
from nested_sampling.samplers.ellipsoidal import EllipsoidConstrainer, MultiEllipsoidConstrainer
import matplotlib.pyplot as plt
from nested_sampling.postprocess import equal_weighted_posterior, marginal_plots
import time

def run_nested(**config):
	ndim = config['ndim']
	
	def priortransform(u):
		assert len(u) == ndim, u
		return u
	if 'seed' in config:
		numpy.random.seed(config['seed'])

	# can use directly
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
		constrainer = FriendsConstrainer(rebuild_every=nlive_points, radial=True, metric = 'euclidean', jackknife=config['jackknife'], force_shrink=config['force_shrink'], verbose=False)
	elif method.startswith('supfriends'): # supreme distance
		constrainer = FriendsConstrainer(rebuild_every=nlive_points, radial=True, metric = 'chebyshev', jackknife=config['jackknife'], force_shrink=config['force_shrink'], verbose=True)
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
		constrainer = MCMCConstrainer(proposer = proposer)
	else:
		raise NotImplementedError('draw_method "%s" not implemented' % method)
	print 'configuring NestedSampler'
	starttime = time.time()
	sampler = NestedSampler(nlive_points = nlive_points, 
		priortransform=priortransform, loglikelihood=loglikelihood, 
		draw_constrained = constrainer.draw_constrained, ndim=ndim)
	constrainer.sampler = sampler
	print 'running nested_integrator to tolerance 0.5'
	assert config['integrator'] == 'normal', config['integrator']
	result = nested_integrator(tolerance=0.5, sampler=sampler, max_samples=2000000)
	endtime = time.time()
	if hasattr(constrainer, 'stats'):
		constrainer.stats()

	output_basename = config['output_basename']
	
	if config.get('seed', 0) == 0:
		x = numpy.array([x for _, x, _ in sampler.samples])
		y = exp([l for _, _, l in sampler.samples])
		plt.plot(x[:,0], y, 'x', color='blue', ms=1)
		plt.savefig(output_basename + 'nested_samples.pdf', bbox_inches='tight')
		plt.close()

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
	
		posterioru, posteriorx = equal_weighted_posterior(result['weights'])
		plt.figure(figsize=(ndim*2, ndim*2))
		marginal_plots(weights=result['weights'], ndim=ndim)
		plt.savefig(output_basename + 'posterior.pdf', bbox_inches='tight')
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
		dict(draw_method='naive'),
		#dict(draw_method='optimizer', constrainer=OptimizeConstrainer()),
		#dict(draw_method='maxfriends'),
		dict(draw_method='radfriends', jackknife=False, force_shrink=False),
		dict(draw_method='supfriends', jackknife=False, force_shrink=False),
		dict(draw_method='radfriends2', jackknife=False, force_shrink=False),
		dict(draw_method='supfriends2', jackknife=False, force_shrink=False),
		dict(draw_method='radfriends-1', jackknife=True, force_shrink=True),
		#dict(draw_method='mcmc-gauss-scale0.1', proposer = 'gauss', adapt=False, scale=0.1),
		#dict(draw_method='mcmc-gauss-scale0.1-adapt', proposer = 'gauss', adapt=True, scale=0.1),
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
	]
]
configs = [dict([[k, v] for d in config for k, v in d.iteritems()]) for config in itertools.product(*configs)]
for c in configs:
	c['algorithm_name'] = 'NS-%s-nlive%d%s' % (c['draw_method'], c['nlive_points'],
		('-' + c['integrator']) if c['integrator'] != 'normal' else '')
	c['run'] = run_nested



