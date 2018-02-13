from __future__ import print_function
import numpy
from numpy import log, exp, pi
import scipy.stats
import sys

difficulty = int(sys.argv[1])
xobs = 0.5+10**-difficulty

# given a certain sigma, compute integral
# exp(-0.5 * ((xobs - x)/sigma)^2) from x=0 to 1

# for 2 xobs:
# exp(-0.5 * (((xobs1 - x)^2 + (xsob2 - x)^2)/sigma^2) from x=0 to 1
# exp(-0.5 * (((0.5+eps - x)^2 + (0.5-eps - x)^2)/sigma^2) from x=0 to 1
# exp(-0.5 * (((eps - x)^2 + (-eps - x)^2)/sigma^2) from x=-0.5 to 0.5
# exp(-0.5 * (((eps^2 - 2*eps*x + x^2 + eps^2 + 2*eps*x + x^2)/sigma^2) from x=-0.5 to 0.5
# exp(-0.5 * (((2*eps^2 + 2*x^2)/sigma^2) from x=-0.5 to 0.5
# exp(-((eps^2 + x^2)/sigma^2)) from x=-0.5 to 0.5


N = 10000
r = []
for i, w in enumerate(numpy.linspace(0, 1, N)):
    sigma = 10**(w*20 - 10)
    a = scipy.stats.norm(xobs, sigma).cdf(0)
    b = scipy.stats.norm(xobs, sigma).cdf(1)
    part = float(b - a)
    r.append(part)
for i in [2,3,5,10]:
    print(i, log(numpy.mean(numpy.array(r)**(i-1))))

