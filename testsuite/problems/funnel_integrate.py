from __future__ import print_function
import numpy
from numpy import log, exp, pi
import sys

difficulty = int(sys.argv[1])
ndim = int(sys.argv[2])
xobs = 0.5+10**-difficulty
def loglikelihood(X, Y):
	width = 10**(X * 20 - 10)
	like = -0.5 * (((Y -xobs)/width)**2 + log(2*pi * width**2))
	return like

N = 10000
M = 100000
x = numpy.linspace(0.5, 1, M)
r = []
for i, w in enumerate(numpy.linspace(0, 1, N)):
    sys.stderr.write("%f ... \r" % w)
    #y = numpy.linspace(0, 1, N)
    #X, Y = numpy.meshgrid(x, y)
    L = loglikelihood(w, x)
    #print L.max(), L.shape
    r.append(numpy.trapz(numpy.exp(L))/len(x))
    if i % 100 == 0:
    	print("%f " % log(numpy.mean(r) + 1e-10)))
print((log(numpy.mean(r)))
#Z = numpy.log(numpy.exp(L - L.max()).mean()) + L.max()
#print Z

