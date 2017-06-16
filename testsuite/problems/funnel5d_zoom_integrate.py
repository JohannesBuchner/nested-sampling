import numpy
from numpy import log, exp, pi
import sys
import scipy.misc

difficulty = int(sys.argv[1])
ndim = int(sys.argv[2])

xobs = numpy.array([0.5+10**-difficulty,0.5-10**-difficulty]*100)
xobs = xobs[:ndim-1].reshape((1,ndim-1))
def loglikelihood(width, x):
	like = -0.5 * (((x - xobs)/width)**2 + log(2*pi * width**2)).sum(axis=1)
	return like

sigma_cut = 10

# compute the area/volume we are actually drawing from
N = 100000
vol_fractions = []
for logsigma in numpy.linspace(0,1,N):
	sigma = 10**(logsigma * 20 - 10)
        # we will only integrate from 0.5 +- sigma_cut * sigma
        if sigma < 10**-difficulty:
		sigma = 10**-difficulty
        width = 2 * sigma * sigma_cut
        integrated_fraction = width
        #print integrated_fraction
        if integrated_fraction > 1:
        	integrated_fraction = 1
        vol_fractions.append(integrated_fraction)
penalty = log(numpy.mean(vol_fractions))
print 'volume penalty:', penalty

#sys.exit()
Ls = []
while True:
	N = 1000000
        # draw samples only in the limited area
        logsigma = numpy.random.uniform(size=(N,1))
        # sigma is probably around 10**-difficulty
        #logsigma = numpy.random.uniform(-difficulty-4, -difficulty+4, size=(N,1))
        # x is probably not more than 10 sigma away from 0.5
        u = numpy.random.uniform(-1, 1, size=(N, ndim-1))
        # we reduce the range from 1 to 0.5 +- width
        sigma = 10**(logsigma * 20 - 10) 
        width = sigma * sigma_cut
        width[sigma < 10**-difficulty] = 10**-difficulty * sigma_cut
        width[width > 0.5] = 0.5
        x = 0.5 + u * width
        # test: draw in a more stupid way
        #x = numpy.random.uniform(size=(N, ndim-1))
        #mask = (numpy.abs(x - 0.5) < width).all(axis=1)
        #x = x[mask,:]
        #sigma = sigma[mask]
        #print mask.mean()*100, exp(penalty)

	L = loglikelihood(sigma, x)
        # lower by penalty
        Ls.append(scipy.misc.logsumexp(L) - log(len(L)) + penalty)
        means = []
        for i in range(20):
        	vals = numpy.exp([Ls[j] for j in numpy.random.randint(0, len(Ls), size=len(Ls))])
                means.append(numpy.log(vals.mean()))
        print '%.5f BS: %.5f +- %.5f (%d x %d)' % (numpy.log(numpy.exp(Ls).mean()), numpy.mean(means), numpy.std(means), len(Ls), N)

