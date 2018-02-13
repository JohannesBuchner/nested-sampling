import numpy
from numpy import log, exp, pi
import sys
import scipy.misc

difficulty = int(sys.argv[1])
ndim = int(sys.argv[2])

def loglikelihood(x):
	width = 10**(x[:,0].reshape(-1,1) * 20 - 10)
	like = -0.5 * (((numpy.abs(x[:,1:] - 0.5) + 10**-difficulty)/width)**2 + log(2*pi * width**2)).sum(axis=1)
        assert len(like) == len(x), (like.shape, x.shape)
	return like

Ls = []
while True:
	N = 1000000
        x = numpy.random.uniform(size=(N, ndim))
	L = loglikelihood(x)
        Ls.append(scipy.misc.logsumexp(L) - log(N))
        means = []
        for i in range(20):
        	vals = numpy.exp([Ls[j] for j in numpy.random.randint(0, len(Ls), size=len(Ls))])
                means.append(numpy.log(vals.mean()))
        print('%.5f BS: %.5f +- %.5f (%d x %d)' % (numpy.log(numpy.exp(Ls).mean()), numpy.mean(means), numpy.std(means), len(Ls), N))

# 0 2 -4.7323
# 1 2 -3.24778
# 2 2 -2.4618
# 3 2 -2.00452
# 4 2 -1.687
# 
# 0 5 -10.463
# 1 5 -5.11407
# 2 5 -3.21154
# 3 5 -2.50263
# 4 5 -2.30399

