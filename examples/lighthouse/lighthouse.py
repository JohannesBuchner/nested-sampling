from __future__ import print_function
"""

Generates data for the light house problem
Analyses it
Makes pretty plots

"""
import numpy
import scipy.stats
from numpy import log, exp, pi
import matplotlib.pyplot as plt
import matplotlib.patches
import sys

N = int(sys.argv[1])

# arrival positions
data = numpy.array([ 4.73,  0.45, -1.73,  1.09,  2.19,  0.12,
	1.31,  1.00,  1.32,  1.07,  0.86, -0.49, -2.59,  1.73,  2.11,
	1.61,  4.98,  1.71,  2.23,-57.20,  0.96,  1.25, -1.56,  2.45,
	1.19,  2.17,-10.66,  1.91, -4.16,  1.92,  0.10,  1.98, -2.51,
	5.55, -0.47,  1.91,  0.95, -0.78, -0.84,  1.72, -0.01,  1.48,
	2.70,  1.21,  4.41, -4.79,  1.33,  0.81,  0.20,  1.58,  1.29,
	16.19,  2.75, -2.38, -1.79,  6.50,-18.53,  0.72,  0.94,  3.64,
	1.94, -0.11,  1.57,  0.57])[:N]

x0 = -1
y0 = 0.5
# sample from cauchy distribution
numpy.random.seed(1)
data = scipy.stats.cauchy(x0, y0).ppf(numpy.random.uniform(0, 1, size=N))

plt.figure()
plt.plot([-2, 2], [0, 0], '-', color='k')
plt.fill_between([-2, 2], [0, 0], [2, 2], color='b', alpha=0.3)
plt.fill_between([-2, 2], [0, 0], [-0.2, -0.2], color='grey')
plt.xlim(-2, 2)
plt.ylim(-0.2, 2)
plt.xticks(numpy.arange(-2, 3))
plt.yticks(numpy.arange(0, 3))

# draw lighthouse at x0, y0
sx, sy = 0.05*2, 0.075*2
w, h = sx/2.*0.98, sy/2.-sy/40.
y = y0 - h
for i in range(4):
	plt.gca().add_patch(matplotlib.patches.Polygon([
		[x0-w, y],
		[x0-(w-sx/2.*0.1), y+h*0.7],
		[x0+w-sx/2.*0.1, y+h*0.7],
		[x0+w, y],
		], color='red' if i%2 == 0 else 'white', linewidth=1))
	y = y + h*0.7
	w, h = w-sx/2.*0.1, h
plt.gca().add_patch(matplotlib.patches.Rectangle((x0-sx/2.*1.2, y0-sy/2.), 
	width=sx*1.2, height=sy/20, color='k', linewidth=1))
plt.gca().add_patch(matplotlib.patches.Rectangle([x0-w*1.2, y], 
	width=w*2*1.2, height=sy/20, color='k', linewidth=1))
y = y+sy/20
plt.gca().add_patch(matplotlib.patches.Rectangle([x0-w, y], 
	width=w*2, height=w*2/(4/2.2),
	facecolor='yellow', edgecolor='black', fill=True, linewidth=1))
y = y+w*2/(4/2.2)
plt.gca().add_patch(matplotlib.patches.Polygon([
	[x0-w*1.2, y],
	[x0, y+w*2/(4/2.2)/3],
	[x0+w*1.2, y]],
	facecolor='k'))

plt.plot(data, numpy.zeros_like(data), 'x ', color='red')
plt.savefig('lighthouse_data.pdf', bbox_inches='tight')

del x0, y0 # forget true values

def loglikelihood(x):
	x0 = x[0]
	y0 = x[1]
	xdist = (data - x0)
	# Cauchy distribution with x0 = x0, gamma = y0
	like = (log(y0) - log(pi) - log(xdist**2 + y0**2)).sum()
	return like

def priortransform(u):
	return numpy.array([u[0] * 4 - 2, u[1] * 2])
# run nested sampling

from nested_sampling.nested_integrator import nested_integrator
from nested_sampling.nested_sampler import NestedSampler
from nested_sampling.samplers.rejection import RejectionConstrainer
from nested_sampling.samplers.friends import FriendsConstrainer
import nested_sampling.postprocess as post
#constrainer = RejectionConstrainer()
constrainer = FriendsConstrainer(radial = True, metric = 'euclidean', jackknife=True)
sampler = NestedSampler(nlive_points = 400, 
	priortransform=priortransform, loglikelihood=loglikelihood, 
	draw_constrained = constrainer.draw_constrained, ndim=2)
constrainer.sampler = sampler
results = nested_integrator(tolerance=0.5, sampler=sampler)

# add contours?

usamples, xsamples = post.equal_weighted_posterior(results['weights'])

u, x, L, width = list(zip(*results['weights']))
x, y = numpy.array(x).T
weight = numpy.add(L, width)
#plt.plot(xsamples[:,0], xsamples[:,1], '.', color='green', alpha=0.1)
#plt.hexbin(x, y, exp(weight - weight.max()), gridsize=40, cmap=plt.cm.RdBu_r, 
#	vmin=0, vmax=1)

#x, y = numpy.array(xsamples).T
#plt.hexbin(x, y, gridsize=40, cmap=plt.cm.RdBu_r, vmax=len(x)/(40.), vmin=0)

# create contours using the lowest values, always summing up until 1%, 10%, 50%
# is contained
z = exp(weight - weight.max()).cumsum()
z /= z.max()

#xi = numpy.linspace(-2, 2, 20)
#yi = numpy.linspace(0, 2, 20)
##from matplotlib.mlab import griddata
##zi = griddata(x, y, z, xi, yi)
#X, Y = numpy.meshgrid(xi, yi)
#Z = X * 0 + Y * 0
#from scipy.interpolate import griddata
#Z = griddata((x, y), z, (X, Y), method='nearest')
#for xj, yj, zj in zip(x, y, z):
#	j = int(((xj + 2) / 4) * len(xi))
#	i = int((yj / 2) * len(yi))
#	Z[i,j] = max(Z[i, j], zj)
import scipy.spatial
for limit, color in [(0.01, 'DarkBlue'), (0.1, 'DeepSkyBlue'), (0.5, 'DarkRed')]:
	mask = z > limit
	zsel = z[mask]
	xsel = x[mask]
	ysel = y[mask]
	hull = scipy.spatial.ConvexHull(numpy.transpose([xsel, ysel]))
	for simplex in hull.simplices:
		plt.plot(xsel[simplex], ysel[simplex], '-', color=color)

#plt.contour(X, Y, Z, [0.01, 0.1, 0.5])



plt.savefig('lighthouse_posterior.pdf', bbox_inches='tight')



