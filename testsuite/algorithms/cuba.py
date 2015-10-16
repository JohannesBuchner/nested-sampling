from __future__ import print_function
"""
Runs the Cuba algorithms: Vegas, Suave, Divonne, Cuhre
"""
import os
# don't use parallelization, because pycuba gets confused, and we parallize runs anyways
os.environ['CUBACORES'] = "1"
import pycuba
import json
import itertools
from math import exp, log
import time

def run_cuba(**config):
	NDIM = config['ndim']
	loglikelihood = config['loglikelihood']
	def Integrand(ndim, xx, ncomp, ff, userdata):
		ff[0] = exp(loglikelihood([xx[i] for i in range(ndim.contents.value)]))
		return 0
	NCOMP = 1

	NNEW = 1000
	FLATNESS = 25.

	KEY1 = 47
	KEY2 = 1
	KEY3 = 1
	MAXPASS = 5
	BORDER = 0.
	MAXCHISQ = 10.
	MINDEVIATION = .25
	NGIVEN = 0
	LDXGIVEN = NDIM
	NEXTRA = 0
	MINEVAL = 100
	MAXEVAL = 2000000
	epsrel=0.5
	epsabs=1e-300
	
	KEY = 0
	
	commonargs = dict(
		mineval=MINEVAL, maxeval=MAXEVAL,
		epsrel=epsrel, epsabs=epsabs, 
		seed = config['seed'] + 1, # 0 stands for random
	)
	
	def print_results(name, results):
		keys = ['nregions', 'neval', 'fail']
		keys = list(filter(results.has_key, keys))
		text = ["%s %d" % (k, results[k]) for k in keys]
		print ("%s RESULT:\t" % name.upper() + "\t".join(text))
		for comp in results['results']:
			print ("%s RESULT:\t" % name.upper() + "%(integral).8f +- %(error).8f\tp = %(prob).3f\n" % comp)
	starttime = time.time()
	if config['cuba_algorithm'] == 'Vegas':
		results = pycuba.Vegas(Integrand, NDIM, verbose=2, **commonargs)
	elif config['cuba_algorithm'] == 'Suave':
		results = pycuba.Suave(Integrand, NDIM, NNEW, FLATNESS, verbose=2 | 4, **commonargs)
	elif config['cuba_algorithm'] == 'Divonne':
		results = pycuba.Divonne(Integrand, NDIM,
			key1=KEY1, key2=KEY2, key3=KEY3, maxpass=MAXPASS,
			border=BORDER, maxchisq=MAXCHISQ, mindeviation=MINDEVIATION,
			ldxgiven=LDXGIVEN, verbose=2, **commonargs)
	elif config['cuba_algorithm'] == 'Cuhre':
		results = pycuba.Cuhre(Integrand, NDIM, key=KEY, verbose=2 | 4, **commonargs)
  	else:
  		assert False, 'Unknown cuba algorithm "%s"!' % config['cuba_algorithm']
  	duration = time.time() - starttime
	print_results(config['cuba_algorithm'], results)
	Z = results['results'][0]['integral']
	Zerr = results['results'][0]['error']

	return dict(
		Z_computed = float(log(abs(Z) + 1e-300)),
		Z_computed_err = float(log(abs(Z+Zerr) + 1e-300) - log(abs(Z) + 1e-300)),
		failed=results['fail'] != 0 and Z >= 0 and Zerr >= 0,
		duration = duration,
	)

configs = [
	dict(cuba_algorithm='Vegas'),
	dict(cuba_algorithm='Suave'),
	dict(cuba_algorithm='Divonne'),
	dict(cuba_algorithm='Cuhre', noseed=True), # deterministic algorithm
]
for c in configs:
	c['algorithm_name'] = 'cuba-%s' % (c['cuba_algorithm'])
	c['run'] = run_cuba


