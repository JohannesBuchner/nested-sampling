from __future__ import print_function
"""
Copyright: Johannes Buchner (C) 2013

Example program to set up nested sampling.

"""
import numpy
from numpy import exp, log, log10, pi
import matplotlib.pyplot as plt

from nested_sampling.nested_integrator import nested_integrator
from nested_sampling.nested_sampler import NestedSampler
from nested_sampling.samplers.rejection import RejectionConstrainer as Constrainer
#from nested_sampling.samplers.affinemcmc import AffineMCMCConstrainer as Constrainer
#from nested_sampling.samplers.svm import SVMConstrainer as Constrainer
#from nested_sampling.samplers.ellipsoidal import EllipsoidConstrainer as Constrainer
#from nested_sampling.samplers.ellipsoidal import MultiEllipsoidConstrainer as Constrainer
from nested_sampling.samplers.friendsnext import FriendsConstrainer


if __name__ == '__main__':
	
	def priortransform(u):
		return u # no transformation at the moment

	def loglikelihood(x):
		a = - 0.5 * ((x - 0.2)/0.05)**2 - 0.5 * log(2*pi*0.05**2)
		b = - 0.5 * ((x - 0.7)/0.05)**2 - 0.5 * log(2*pi*0.05**2)
		return log(exp(a) + exp(b) + 0.01e-100)

	mu = 0.5
	sigma = 0.05
	def loglikelihood(x):
		a = -0.5 * (((x[0] - 0.5)/0.03)**2)
		b = -0.5 * (((x[1] - 0.5)/0.03)**2)
		return a + b - 0.5 * log(2*pi) * 2 - 0.5 * log(0.03**2) * 2
	# evidence should be log(2.01) = 0.698
	logZanalytic = log(0.01 + (2*pi)**0.5 / 10)
	logZanalytic = log(2.01)
	logZanalytic = 0
	
	constrainer = Constrainer()
	constrainer = FriendsConstrainer(rebuild_every=50, radial=True, metric='euclidean', jackknife=False, force_shrink=False, verbose=False)
	
	print('preparing sampler')
	sampler = NestedSampler(nlive_points = 400, priortransform=priortransform, loglikelihood=loglikelihood, 
		draw_constrained = constrainer.draw_constrained, ndim=2)
	# tell constrainer about sampler so they can interact
	constrainer.sampler = sampler
	print('running sampler')
	result = nested_integrator(tolerance=0.2, sampler=sampler)

	x = numpy.array([x for _, x, _ in sampler.samples])
	y = numpy.exp([l for _, _, l in sampler.samples])
	plt.plot(x, y, 'x', color='blue', ms=1)
	plt.savefig('nested_samples.pdf', bbox_inches='tight')
	plt.close()
	
	weights = result['weights']
	L = numpy.array([Li for ui, xi, Li, logwidth in weights])
	widths = numpy.array([logwidth for ui, xi, Li, logwidth in weights])
	plt.plot(exp(widths), exp(L), 'x-', color='blue', ms=1)
	plt.xlabel('prior mass')
	plt.ylabel('likelihood')
	plt.xscale('log')
	plt.yscale('log')
	#plt.xlim(0, 1)
	plt.savefig('nested_integral.pdf', bbox_inches='tight')
	plt.close()

	print('analytic logZ:', logZanalytic)
	u = numpy.linspace(0, 1, len(sampler.samples) - sampler.nlive_points)
	x = priortransform(u)
	L = loglikelihood(x)
	print('monte carlo integration (%d samples) logZ:' % len(u), log(exp(L).mean()))
	u = numpy.linspace(0, 1, len(sampler.samples))
	x = priortransform(u)
	L = loglikelihood(x)
	print('monte carlo integration (%d samples) logZ:' % len(u), log(exp(L).mean()))

	print('nested sampling (%d samples) logZ = ' % len(result['samples']), result['logZ'], result['logZerr'])



