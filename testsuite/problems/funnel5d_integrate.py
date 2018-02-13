from __future__ import print_function
import numpy
from numpy import log, exp, pi
import sys
import scipy.misc

difficulty = int(sys.argv[1])
ndim = int(sys.argv[2])

xobs = numpy.array([0.5+10**-difficulty,0.5-10**-difficulty]*100)
xobs = xobs[:ndim-1].reshape((1,ndim-1))
def loglikelihood(x):
	width = 10**(x[:,0].reshape(-1,1) * 20 - 10)
	like = -0.5 * (((x[:,1:] - xobs)/width)**2 + log(2*pi * width**2)).sum(axis=1)
	assert len(like) == len(x), (like.shape, x.shape)
	return like

Ls = []
while True:
	N = 1000000
	x = numpy.random.uniform(size=(N, ndim))
	L = loglikelihood(x)
	Ls.append(scipy.misc.logsumexp(L) - log(N))
	#print '%.4f +- %.4f (%d)' % (numpy.mean(Ls), numpy.std(Ls), len(Ls))
	#print '%.4f +- %.4f (%d x %d)' % (numpy.log(numpy.exp(Ls).mean()), numpy.log(numpy.exp(Ls).std()), len(Ls), N)
	means = []
	for i in range(20):
		vals = numpy.exp([Ls[j] for j in numpy.random.randint(0, len(Ls), size=len(Ls))])
		means.append(numpy.log(vals.mean()))
	print('%.5f BS: %.5f +- %.5f (%d x %d)' % (numpy.log(numpy.exp(Ls).mean()), numpy.mean(means), numpy.std(means), len(Ls), N))

