"""
Runs Nestle
"""
import itertools
import nestle
import numpy
from numpy import log, exp
import sys
import time

def run_nestle(**config):
	ndim = config['ndim']
	
	def priortransform(u):
		assert len(u) == ndim, u
		return u
	def dump_callback(info):
		sys.stderr.write("\r%d|%d|logz=%.4f|eff=%f%%    " % (info['it'], info['ncall'], info['logz'], info['it']*100./info['ncall']))
		
		#if info['it'] % 50 != 0: return
		#print "Replacements: %d" % (info['ncall'])
		#print "Samples: %d" % (info['it'])
		#print "Efficiency: %f" % (info['ncall']/info['it'])
		#print "Nested Sampling ln(Z): %f" % (info['logz'])
	if 'seed' in config:
		numpy.random.seed(config['seed'])
	
	# can use directly
	loglikelihood = config['loglikelihood']
	nlive_points = config['nlive_points']
	method = config['method']
	if config.get('unlimited_sampling', False):
		max_samples = None
	else:
		max_samples = 2000000
	print
	print 'running nestle ...'
	options = dict()
	#if 'enlarge' in config:
	#	options['enlarge'] = config['enlarge']
	starttime = time.time()
	result = nestle.sample(loglikelihood=loglikelihood, prior_transform=priortransform, ndim=ndim, npoints=nlive_points,
		method=method, update_interval=None, maxcall=max_samples, dlogz=0.5, rstate=numpy.random, callback=dump_callback,
		**options)
	endtime = time.time()
	output_basename = config['output_basename']
	print
        print 'nestle done lnZ = %(logz).1f +- %(logzerr).1f' % (result)
	
	if config.get('seed', 0) == 0:
		import matplotlib.pyplot as plt
		x = result['samples']
		y = exp(result['logl'])
		plt.plot(x[:,0], y, 'x', color='blue', ms=1)
		plt.savefig(output_basename + 'nested_samples.pdf', bbox_inches='tight')
		plt.close()

		L = result['logl']
		width = result['weights']
		plt.plot(width, L, 'x-', color='blue', ms=1, label='Z=%.2f (%.2f)' % (
			result['logz'], log(exp(L + width).sum())))
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
	
		#posterioru, posteriorx = equal_weighted_posterior(result['weights'])
		#plt.figure(figsize=(ndim*2, ndim*2))
		#marginal_plots(weights=result['weights'], ndim=ndim)
		#plt.savefig(output_basename + 'posterior.pdf', bbox_inches='tight')
		#plt.close()

	return dict(
		Z_computed = float(result['logz']),
		Z_computed_err = float(result['logzerr']),
		niterations = result['niter'],
		duration = endtime - starttime,
	)

configs = [
	[
		dict(nlive_points=100),
		dict(nlive_points=400),
		dict(nlive_points=1000),
	], [
		dict(method='classic'),
		dict(method='single'),
		dict(method='multi'),
		#dict(method='single-robust'),
		#dict(method='multi-robust'),
		#dict(method='single-veryrobust'),
		#dict(method='multi-veryrobust'),
		#dict(method='multi-limitedrobust'),
		#dict(method='multi-simplelimitedrobust'),
		dict(method='multi-rememberingrobust'),
	]
]
configs = [dict([[k, v] for d in config for k, v in d.iteritems()]) for config in itertools.product(*configs)]
for c in configs:
	c['algorithm_name'] = 'nestle-%s-nlive%d' % (c['method'], c['nlive_points'])
	c['run'] = run_nestle



