import numpy
from numpy import log, exp, pi
import sys

difficulty = int(sys.argv[1])
ndim = int(sys.argv[2])
xobs = 0.5+10**-difficulty
def loglikelihood(X, Y):
	width = 10**(X * 20 - 10)
	like = -0.5 * (((numpy.abs(Y - 0.5) + 10**-difficulty)/width)**2 + log(2 * pi * width**2))
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
    	sys.stdout.write("\r%f " % log(numpy.mean(r) + 1e-10))
sys.stdout.write((' '*40) +  '\r')
print(difficulty, ndim, log(numpy.mean(r)))

# 0 2 -4.73259517967                                
# 1 2 -3.24667664298                                
# 2 2 -2.46077086296                                
# 3 2 -2.00265027381                                
# 4 2 -1.68773994179                                
# 5 2 -1.44661682682                                


