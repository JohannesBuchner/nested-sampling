"""
Runs the MultiNest algorithm
"""
import pymultinest
import json
import itertools
import os, sys
import glob
import time
keep_results = False
plot = True

def run_multinest(**config):
	n_params = config['ndim']
	output_basename = config['output_basename']
	# we use a flat prior
        def myprior(cube, ndim, nparams):
                pass
	loglikelihood = config['loglikelihood']	
	def myloglike(cube, ndim, nparams):
		try:
			l = loglikelihood([cube[i] for i in range(ndim)])
		except Exception as e:
			print 'ERROR:', type(e), e
			sys.exit(-127)
		return l
	nlive_points = config['nlive_points']
	if config.get('unlimited_sampling', False):
		max_samples = 0
	else:
		max_samples = 2000000
	mn_args = dict(
		importance_nested_sampling = config['importance_nested_sampling'],
		outputfiles_basename = output_basename + 'out_',
		resume = False,
		verbose = True,
		n_params = n_params,
		n_live_points = nlive_points,
		sampling_efficiency = 'model',
		const_efficiency_mode = False,
		evidence_tolerance = 0.5,
		seed = config['seed'],
		max_iter = max_samples,
	)
	starttime = time.time()
	pymultinest.run(myloglike, myprior, mn_args['n_params'], **mn_args)
	duration = time.time() - starttime
	with file('%sparams.json' % mn_args['outputfiles_basename'], 'w') as f:
		parameters = ['%d' % (i+1) for i in range(mn_args['n_params'])]
		json.dump(parameters, f, indent=2)
	a = pymultinest.Analyzer(n_params = mn_args['n_params'],
		outputfiles_basename = mn_args['outputfiles_basename'])
	s = a.get_stats()
	results = dict(
		Z_computed = s['global evidence'],
        	Z_computed_err = s['global evidence error'],
        	duration = duration,
        )
        # store marginal plot for debugging
	if plot and config['seed'] == 0:
		import matplotlib.pyplot as plt
		p = pymultinest.PlotMarginalModes(a)
		plt.figure(figsize=(5*n_params, 5*n_params))
		for i in range(n_params):
			plt.subplot(n_params, n_params, n_params * i + i + 1)
			p.plot_marginal(i, with_ellipses = False, with_points = False, grid_points=50)
			plt.ylabel("Probability")
			plt.xlabel(parameters[i])
		
			for j in range(i):
				plt.subplot(n_params, n_params, n_params * j + i + 1)
				p.plot_conditional(i, j, with_ellipses = False, with_points = False, grid_points=30)
				plt.xlabel(parameters[i])
				plt.ylabel(parameters[j])
				plt.ylim(0, 1)
				plt.xlim(0, 1)

		plt.savefig('%smarg.png' % output_basename, bbox_inches='tight')
		plt.close()
	
	if config['seed'] != 0 and not keep_results:
		# destroy the evidence (to save disk space)
		for f in glob.iglob(mn_args['outputfiles_basename'] + "*"):
			print 'deleting %s' % f
			os.remove(f)
	
        return results

configs = [
	[
		dict(nlive_points=100),
		dict(nlive_points=400),
		dict(nlive_points=1000),
	], [
		dict(importance_nested_sampling=False), 
		dict(importance_nested_sampling=True)
	]
]
configs = [dict([[k, v] for d in config for k, v in d.iteritems()]) for config in itertools.product(*configs)]
for c in configs:
	c['algorithm_name'] = 'multinest-nlive%d%s' % (c['nlive_points'], '-INS' if c['importance_nested_sampling'] else '')
	c['run'] = run_multinest



