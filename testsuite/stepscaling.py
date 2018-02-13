import testbase
import problems.loggamma
import problems.gauss
import algorithms.nest
import algorithms.multinest
import algorithms.runnestle
import re
import numpy
import matplotlib.pyplot as plt
import sys

# here we test how many MH steps are needed to solve a given problem
# assuming a correct filter function

# specify selected algorithms
algorithms_generators = [
	lambda nlive, nrepeats: dict(draw_method='hmlfriendsTM-harm+%dsteps' % nrepeats, 
		metriclearner='truncatedmahalanobis', 
		keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True,
		proposer = 'harm', nsteps=nrepeats, nminaccepts=nrepeats, 
		integrator='normal-bs',
		algorithm_name='NS-hmlfriendsTM-switch%d-harm+%dsteps-nlive%d-normal-bs' % (100+nrepeats*5, nrepeats, nlive),
		algorithm_shortname='NS-hmlfriendsTM-switch-harm',
		switchover_efficiency=1./(100 + nrepeats*5),
		nlive_points = nlive,
		run = algorithms.nest.run_nested,
		color='g', ls='-', marker='s',
		disable=True, # with switchover
		),
	lambda nlive, nrepeats: dict(draw_method='hmlfriendsTM-harm+%dsteps' % nrepeats, 
		metriclearner='truncatedmahalanobis', 
		keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True,
		proposer = 'harm', nsteps=nrepeats, nminaccepts=nrepeats, 
		integrator='normal-bs',
		algorithm_name='NS-hmlfriendsTM-harm+%dsteps-nlive%d-normal-bs' % (nrepeats, nlive),
		algorithm_shortname='NS-hmlfriendsTM-harm',
		switchover_efficiency=0,
		nlive_points = nlive,
		run = algorithms.nest.run_nested,
		color='g', ls='-', marker='s',
		),
	lambda nlive, nrepeats: dict(draw_method='hmlfriendsTM-ess+%dsteps' % nrepeats, 
		metriclearner='truncatedmahalanobis', 
		keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True,
		proposer = 'ess', nsteps=nrepeats, nminaccepts=nrepeats, 
		integrator='normal-bs',
		algorithm_name='NS-hmlfriendsTM-switch%d-ess+%dsteps-nlive%d-normal-bs' % (100+nrepeats*5, nrepeats, nlive),
		algorithm_shortname='NS-hmlfriendsTM-switch-ess',
		switchover_efficiency=1./(100 + nrepeats*5),
		nlive_points = nlive,
		run = algorithms.nest.run_nested,
		color='orange', ls='-', marker='o',
		disable=True, # with switchover
		),
	lambda nlive, nrepeats: dict(draw_method='hmlfriendsTM-ess+%dsteps' % nrepeats, 
		metriclearner='truncatedmahalanobis', 
		keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True,
		proposer = 'ess', nsteps=nrepeats, nminaccepts=nrepeats, 
		integrator='normal-bs',
		algorithm_name='NS-hmlfriendsTM-ess+%dsteps-nlive%d-normal-bs' % (nrepeats, nlive),
		algorithm_shortname='NS-hmlfriendsTM-ess',
		switchover_efficiency=0,
		nlive_points = nlive,
		run = algorithms.nest.run_nested,
		color='orange', ls='-', marker='o',
		disable=True,
		),
]

problem_generators = {
	'loggammaI_multimodal': lambda ndim: problems.loggamma.create_problem_loggammaI_multimodal(ndim=ndim, problem_name='loggammaI_multimodal%dd' % ndim),
	'gauss': lambda ndim: problems.gauss.create_problem_gauss(ndim=ndim, problem_name='gauss%dd' % ndim),
	'thinshell': lambda ndim: problems.gauss.create_problem_shell(ndim=ndim, problem_name='shell%dd-thin' % ndim, width=[0.001/12*16./ndim]*2),
	#'multigauss': lambda ndim: problems.gauss.create_problem_shell(ndim=ndim, problem_name='multigauss%dd' % ndim),
}

problemname = sys.argv[1]
problem_generator = problem_generators[problemname]
ndim = int(sys.argv[2])

# run the algorithm for d=1,2,4,etc.
# on a single problem
#dims = [16] #, 64, 128

"""
Switchover occurs at
16d: 1500/1000 iterations for gauss/thinshell
"""

problem = problem_generator(ndim)
plt.figure(figsize=(6, 6))
for i, algorithm_generator in list(enumerate(algorithms_generators)): #[::-1]:
	results = []
	true_value = []
	nrepeats_selected = []
	nevals = []
	nlive_points = max(400, 25 * ndim)
	last_was_correct = False
	for nrepeats in [1, 2, 4, 8, 16, 32]:
		if nrepeats > 1 + ndim:
			continue
		# run algorithm
		algorithm = algorithm_generator(nlive_points, nrepeats)
		if algorithm.get('disable', False): 
			continue
		algorithm['unlimited_sampling'] = True
		color = algorithm['color']
		ls = algorithm['ls']
		marker = algorithm['marker']
		lw = 2 if ls == '-' else 1
		name = algorithm['algorithm_name']
		print('preparing...', name, 'against', problem['problem_name'])
		result = testbase.run(problem, algorithm, seed=1)
		if 'normal-bs' in name and result['Z_computed_err'] > 0.5:
			# re-bootstrapping quotes too large errors
			result['Z_computed_err'] = 0.5
		print(algorithm['algorithm_name'], result['neval'], result['Z_computed'], result['Z_computed_err'])
		true_value.append(problem['Z_analytic'])
		nrepeats_selected.append(nrepeats)
		results.append(result)
		this_is_correct = abs(result['Z_computed'] - problem['Z_analytic']) < result['Z_computed_err']
		if last_was_correct and this_is_correct:
			print('previous scaling and this scaling are correct, so we are done.')
			break
		last_was_correct = this_is_correct
	if len(nrepeats_selected) == 0:
		continue
	plt.errorbar(x=numpy.array(nrepeats_selected) * 1.02**(i - 2), 
		y=[r['Z_computed']-t for r, t in zip(results, true_value)], 
		yerr=[r['Z_computed_err'] for r in results],
		marker=marker, linestyle=ls, color=color, lw=lw, ms=4,
		label=algorithm['algorithm_shortname'],
		)
	true_value = [0] * len(nrepeats_selected)
	plt.plot(nrepeats_selected, true_value, '-', lw=3, color='k')

plt.legend(loc='best', prop=dict(size=8))
#plt.xlim(min(dims)/1.5, max(dims)*1.5)
#plt.xscale('log')
plt.ylabel('ln(Z)')
plt.xlabel('Number of Steps')
#plt.xticks(dims, dims)
ylo, yhi = plt.ylim()
ylo = min(ylo, -3)
yhi = max(yhi, +3)
plt.ylim(ylo, yhi)
plt.savefig('stepscaling_%s.pdf' % problem['problem_name'], bbox_inches='tight')
plt.close()

