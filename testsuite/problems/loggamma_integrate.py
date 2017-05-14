import numpy
import sys

hardness = int(sys.argv[1])
r = 1
rerr = 0.1 / hardness
prod = 0.4
proderr = 0.5 / hardness**0.5

def loglikelihood(X, Y):
	x = X * 4 - 2
	y = Y * 4 - 2
	partring = ((((x**2 + y**2)**0.5 - r)/rerr)**2)
	partx = (((y - prod)/proderr)**2)
	return -0.5 * (partring + partx)

N = 4000
x = numpy.linspace(0, 1, N)
y = numpy.linspace(0, 1, N)
X, Y = numpy.meshgrid(x, y)
L = loglikelihood(X, Y)
print L.max(), L.shape
Z = numpy.log(numpy.exp(L - L.max()).mean()) + L.max()
print Z

