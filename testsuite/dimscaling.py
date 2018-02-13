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

# specify selected algorithms
#[for c in algorithms.nest.config if 'NS-hmlmultiellipsoid-harm+1steps-nlive400-' in c['algorithm_name'] ]
algorithms_generators = [
	#dict(draw_method='hmlmultiellipsoid-harm+1steps', metriclearner='simplescaling', 
	#	proposer = 'harm', nsteps=1, nminaccepts=1, 
	#	integrator='normal-bs',
	#	algorithm_name='NS-hmlmultiellipsoid-harm+1steps-nlive%d-normal-bs',
	#	run = algorithms.nest.run_nested
	#	),
	#lambda nlive, _: dict(
	#	importance_nested_sampling=False,
	#	algorithm_name='multinest-nlive%d' % (nlive),
	#	algorithm_shortname='multinest',
	#	nlive_points = nlive,
	#	run = algorithms.multinest.run_multinest,
	#	color='yellow', ls='--',
	#	),
	#lambda nlive, _: dict(
	#	importance_nested_sampling=True,
	#	algorithm_name='multinest-nlive%d-INS' % (nlive),
	#	algorithm_shortname='multinest-INS',
	#	nlive_points = nlive,
	#	run = algorithms.multinest.run_multinest,
	#	color='orange', ls='-',
	#	),
	lambda nlive, _: dict(
		algorithm_name='nestle-nlive%d' % (nlive),
		algorithm_shortname='multiellipsoid-20%enlarge',
		nlive_points = nlive,
		run = algorithms.runnestle.run_nestle,
		method='multi',
		color='gray', ls=':', marker='',
		),
	lambda nlive, _: dict(
		algorithm_name='nestle-remembering-nlive%d' % (nlive),
		algorithm_shortname='multiellipsoid-robustenlarge',
		nlive_points = nlive,
		run = algorithms.runnestle.run_nestle,
		method='multi-rememberingrobust',
		color='orange', ls='-', marker='o',
		#disable=True, #running in another thread
		),
	lambda nlive, nrepeats: dict(draw_method='hmlfriendsM-harm+%dsteps' % nrepeats, 
		metriclearner='mahalanobis', 
		keep_phantom_points=False, optimize_phantom_points=False, force_shrink=False,
		unfiltered=True,
		proposer = 'harm', nsteps=nrepeats, nminaccepts=nrepeats, 
		integrator='normal-verysmall',
		algorithm_name='NS-ml-harm+%dsteps-nlive%d-normal-verysmall' % (nrepeats, nlive),
		algorithm_shortname='harm-ml',
		nlive_points = nlive,
		run = algorithms.nest.run_nested,
		color='brown', ls='-', marker='x',
		),
	lambda nlive, nrepeats: dict(draw_method='hmlfriendsM-harm+varsteps', 
		metriclearner='mahalanobis', 
		keep_phantom_points=False, optimize_phantom_points=False, force_shrink=False,
		unfiltered=True,
		proposer = 'harm', nsteps=-2, nminaccepts=2, 
		integrator='normal-verysmall',
		algorithm_name='NS-ml-harm+varsteps-nlive%d-normal-verysmall' % (nlive),
		algorithm_shortname='harm-ml',
		nlive_points = nlive,
		run = algorithms.nest.run_nested,
		color='brown', ls='-', marker='+',
		disable=True, # just too slow
		),
	#lambda nlive, nrepeats: dict(draw_method='hmlmultiellipsoid2-harm+%dsteps' % nrepeats, 
	#	metriclearner='simplescaling', 
	#	proposer = 'harm', nsteps=nrepeats, nminaccepts=nrepeats, 
	#	integrator='normal-bs',
	#	algorithm_name='NS-hmlmultiellipsoid2-harm+%dsteps-nlive%d-normal-bs' % (nrepeats, nlive),
	#	algorithm_shortname='NS-hmlmultiellipsoid2-harm',
	#	nlive_points = nlive,
	#	enlarge=2,
	#	run = algorithms.nest.run_nested,
	#	color='b', ls=':',
	#	),
	#lambda nlive, nrepeats: dict(draw_method='hmlmultiellipsoid2-harm+%dsteps' % nrepeats, 
	#	metriclearner='simplescaling', 
	#	proposer = 'harm', nsteps=nrepeats, nminaccepts=nrepeats, 
	#	integrator='normal-bs',
	#	algorithm_name='NS-hmlmultiellipsoid2-switch%d-harm+%dsteps-nlive%d-normal-bs' % (100+nrepeats*5, nrepeats, nlive),
	#	algorithm_shortname='NS-hmlmultiellipsoid2-switch-harm',
	#	switchover_efficiency=1./(100 + nrepeats*5),
	#	nlive_points = nlive,
	#	enlarge=2,
	#	run = algorithms.nest.run_nested,
	#	color='b', ls='-',
	#	),

	lambda nlive, nrepeats: dict(draw_method='hmlmultiellipsoidBS-harm+%dsteps' % nrepeats, 
		metriclearner='simplescaling', 
		proposer = 'harm', nsteps=nrepeats, nminaccepts=nrepeats, 
		integrator='normal-bs',
		algorithm_name='NS-hmlmultiellipsoidBS-harm+%dsteps-nlive%d-normal-bs' % (nrepeats, nlive),
		algorithm_shortname='multiellipsoid-harm-ml',
		nlive_points = nlive,
		bs_enabled=True,
		run = algorithms.nest.run_nested,
		color='b', ls=':', marker='o',
		disable=True, # only enable when trying to show scaling without filter
		),
	lambda nlive, nrepeats: dict(draw_method='hmlmultiellipsoidBSM-harm+%dsteps' % nrepeats, 
		metriclearner='mahalanobis', 
		proposer = 'harm', nsteps=nrepeats, nminaccepts=nrepeats, 
		integrator='normal-bs',
		algorithm_name='NS-hmlmultiellipsoidBSM-switch%d-harm+%dsteps-nlive%d-normal-bs' % (100+nrepeats*5, nrepeats, nlive),
		algorithm_shortname='multiellipsoid-switch-harm-ml',
		switchover_efficiency=1./(100 + nrepeats*5),
		nlive_points = nlive,
		bs_enabled=True,
		enlarge=2,
		run = algorithms.nest.run_nested,
		color='b', ls='-', marker='o',
		),

	lambda nlive, nrepeats: dict(draw_method='hmlfriendsTM-harm+%dsteps-optphantoms' % nrepeats, 
		metriclearner='truncatedmahalanobis', 
		keep_phantom_points=True, optimize_phantom_points=True, force_shrink=True,
		proposer = 'harm', nsteps=nrepeats, nminaccepts=nrepeats, 
		integrator='normal-bs',
		algorithm_name='NS-hmlfriendsTM-harm+%dsteps-optphantoms-nlive%d-normal-bs' % (nrepeats, nlive),
		algorithm_shortname='NS-hmlfriendsTM-harm',
		nlive_points = nlive,
		run = algorithms.nest.run_nested,
		color='g', ls=':', marker='s',
		disable=True, # we do not need phantom points
		),
	lambda nlive, nrepeats: dict(draw_method='hmlfriendsTM-harm+%dsteps-optphantoms' % nrepeats, 
		metriclearner='truncatedmahalanobis', 
		keep_phantom_points=True, optimize_phantom_points=True, force_shrink=True,
		proposer = 'harm', nsteps=nrepeats, nminaccepts=nrepeats, 
		integrator='normal-bs',
		algorithm_name='NS-hmlfriendsTM-switch%d-harm+%dsteps-optphantoms-nlive%d-normal-bs' % (100+nrepeats*5, nrepeats, nlive),
		algorithm_shortname='radfriends-switch-harm-Tml',
		switchover_efficiency=1./(100 + nrepeats*5),
		nlive_points = nlive,
		run = algorithms.nest.run_nested,
		color='g', ls='-', marker='s',
		disable=True, # we do not need phantom points
		),
	lambda nlive, nrepeats: dict(draw_method='hmlfriendsTM-harm+%dsteps' % nrepeats, 
		metriclearner='truncatedmahalanobis', 
		keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True,
		proposer = 'harm', nsteps=nrepeats, nminaccepts=nrepeats, 
		integrator='normal-bs',
		algorithm_name='NS-hmlfriendsTM-harm+%dsteps-nlive%d-normal-bs' % (nrepeats, nlive),
		algorithm_shortname='radfriends-harm-Tml',
		nlive_points = nlive,
		run = algorithms.nest.run_nested,
		color='g', ls=':', marker='s',
		disable=True, # only enable when trying to show scaling without filter
		),
	lambda nlive, nrepeats: dict(draw_method='hmlfriendsTM-harm+%dsteps' % nrepeats, 
		metriclearner='truncatedmahalanobis', 
		keep_phantom_points=False, optimize_phantom_points=False, force_shrink=True,
		proposer = 'harm', nsteps=nrepeats, nminaccepts=nrepeats, 
		integrator='normal-bs',
		algorithm_name='NS-hmlfriendsTM-switch%d-harm+%dsteps-nlive%d-normal-bs' % (100+nrepeats*5, nrepeats, nlive),
		algorithm_shortname='radfriends-switch-harm-Tml',
		switchover_efficiency=1./(100 + nrepeats*5),
		nlive_points = nlive,
		run = algorithms.nest.run_nested,
		color='g', ls='-', marker='s',
		),
]

problem_generators = {
	'loggammaI_multimodal': lambda ndim: problems.loggamma.create_problem_loggammaI_multimodal(ndim=ndim, problem_name='loggammaI_multimodal%dd' % ndim),
	'gauss': lambda ndim: problems.gauss.create_problem_gauss(ndim=ndim, problem_name='gauss%dd' % ndim),
	'shell': lambda ndim: problems.gauss.create_problem_shell(ndim=ndim, problem_name='shell%dd' % ndim),
	'thinshell': lambda ndim: problems.gauss.create_problem_shell(ndim=ndim, problem_name='shell%dd-thin' % ndim, width=[0.001/12*16./ndim]*2),
	#'multigauss': lambda ndim: problems.gauss.create_problem_shell(ndim=ndim, problem_name='multigauss%dd' % ndim),
}

problemname = sys.argv[1]
problem_generator = problem_generators[problemname]
maxlogdim = int(sys.argv[2])

# run the algorithm for d=1,2,4,etc.
# on a single problem
dims = []
for i in range(10):
	if i > maxlogdim:
		break
	dims.append(2**i)
	#if i > 0:
	#	dims.append(int(round(2**(i*0.5))))
	#dims.append(2**i+1)
#dims.append(100)
dims = sorted(set(dims))


plt.figure(figsize=(6, 14))
for i, algorithm_generator in list(enumerate(algorithms_generators)): #[::-1]:
	results = []
	true_value = []
	dims_selected = []
	nevals = []
	for ndim in dims:
		try:
			problem = problem_generator(ndim)
		except AssertionError as e:
			# dimensionality may be too low for the problem
			print 'skipping ndim=%s, because:' % ndim, e
			continue
		#nlive_points = 25 * ndim
		nlive_points = max(400, 25 * ndim)
		nrepeats = 1 + ndim
		#if nrepeats > 20:
		#	nrepeats = 20
		#nrepeats = 1 + ndim * 2
		#if ndim > 20 and i < 4:
		#	nrepeats = 1 + ndim*2
		#nrepeats = 1 + ndim * 3
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
		print 'preparing...', name, 'against', problem['problem_name']
		if name.startswith('NS-h') and 'switch' not in name and ndim > 16:
			# no need to run these algorithms
			print 'not running', name
			continue
		if 'multinest' in name and ndim >= 64:
			print 'not running', name
			continue
		if 'nestle' in name and ndim > 100:
			print 'not running', name
			continue
		result = testbase.run(problem, algorithm, seed=1)
		if 'normal-bs' in name and result['Z_computed_err'] > 0.5:
			# re-bootstrapping quotes too large errors
			result['Z_computed_err'] = 0.5
		print algorithm['algorithm_name'], result['neval'], result['Z_computed'], result['Z_computed_err']
		true_value.append(problem['Z_analytic'])
		dims_selected.append(ndim)
		results.append(result)
		if False and abs(result['Z_computed'] - problem['Z_analytic']) > 3 * max(0.5, result['Z_computed_err']):
			# wrong!
			pass
		else:
			nevals.append(result['neval'])
		if abs(result['Z_computed'] - problem['Z_analytic']) > 5 * max(0.5, result['Z_computed_err']):
			# very very wrong! lets stop.
			break
		if result['neval'] > 20e6:
			# already using a lot of evaluations, lets not go further
			print 'already using a lot of evaluations, lets not go further'
			break
	if len(dims_selected) == 0: continue
	plt.subplot(2, 1, 1)
	plt.errorbar(x=numpy.array(dims_selected) * 1.02**(i - 2), 
		y=[r['Z_computed']-t for r, t in zip(results, true_value)], 
		yerr=[r['Z_computed_err'] for r in results],
		marker=marker, linestyle=ls, color=color, lw=lw, ms=4,
		label=algorithm['algorithm_shortname'],
		)
	plt.subplot(2, 1, 2)
	plt.plot(dims_selected[:len(nevals)], nevals, 
		marker=marker, linestyle=ls, color=color, lw=lw, ms=4,
		label=algorithm['algorithm_shortname'],
		)
plt.subplot(2, 1, 2)
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Number of evaluations')
plt.xlabel('Dimensionality')
plt.xticks(dims, dims)
plt.xlim(min(dims)/1.5, max(dims)*1.5)
plt.legend(loc='best', prop=dict(size=8))
plt.subplot(2, 1, 1)
plt.legend(loc='best', prop=dict(size=8))
true_value = [0] * len(dims)
#plt.plot(dims_selected, true_value, '-', lw=3, color='k')
plt.plot(dims, true_value, '-', lw=3, color='k')
plt.xlim(min(dims)/1.5, max(dims)*1.5)
plt.xscale('log')
plt.ylabel('ln(Z)')
plt.xlabel('Dimensionality')
plt.xticks(dims, dims)
ylo, yhi = plt.ylim()
ylo = min(ylo, min(true_value) - 3)
yhi = max(yhi, max(true_value) + 3)
plt.ylim(ylo, yhi)
plt.ylim(ylo, yhi)
plt.savefig('dimscaling_%s.pdf' % problemname, bbox_inches='tight')
plt.close()

