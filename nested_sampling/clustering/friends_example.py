from __future__ import print_function
import numpy
import scipy.spatial, scipy.cluster
import numpy
from numpy import logical_and, log
import matplotlib.pyplot as plt
import sys
# rectangle
def rect(x):
	return logical_and(x > 0.4, x < 0.6).all() and numpy.random.uniform() > x[0]

# line
def line(x):
	return numpy.abs(x[0] - x[1]) < 0.04 and x[0] > 0.2 and x[0] < 0.8 #and numpy.random.uniform() > x[0]
# banana
def banana(x):
	return numpy.abs(x[1] - x[0]**10.) < 0.1 and x[0] > 0.6 and x[0] < 0.9 #and numpy.random.uniform() > x[0]

numpy.random.seed(0)
coords = numpy.random.uniform(0, 1, size=10000).reshape((-1, 2))
plt.figure(figsize=(10, 6))
for i, shape in enumerate([rect, line, banana]):
	plt.subplot(2, 3, i+1)
	# sample with rejections
	mask = numpy.array([shape(c) for c in coords])
	chosen = coords[mask]
	print( 'chosen:', i, mask.sum())
	#chosen = chosen[:100,:]
	plt.plot(chosen[:,0], chosen[:,1], '.')
	plt.xlim(0, 1); plt.ylim(0, 1)

	# compute neighbor-neighbor distances
	dist = numpy.zeros((len(chosen), len(chosen), 2))
	knearest = []
	for j, a in enumerate(chosen):
		nearest = (None, None)
		for k, b in enumerate(chosen):
			#dist[j, k, :] = numpy.max(a - b, axis=0)
			if k == j:
				continue
			d = scipy.spatial.distance.euclidean(a, b)
			if nearest[0] is None or d < nearest[0]:
				nearest = (d, k)
		b = chosen[nearest[1]]
		#plt.plot([a[0], b[0]], [a[1], b[1]], '-', color='k')

	# spatial clustering
	distances = scipy.spatial.distance.cdist(chosen, chosen)
	cluster = scipy.cluster.hierarchy.single(distances)
	
	# create distance histogram
	clusterdists = cluster[:,2]
	
	# create k nearest neighbors distance histogram
	knearest = [d.argsort()[1:1+2*2] for d in distances]
	for a, k in zip(chosen, knearest):
		for b in chosen[k]:
			#print a, b
			plt.plot([a[0], b[0]], [a[1], b[1]], '-', color='r')
		break
	##kdists = [numpy.max(chosen[knearest], axis=0), a for j, a in enumerate(knearest)]
	#kdists = [numpy.max([scipy.spatial.distance.euclidean(b, chosen[j,:]) for b in chosen[k]], axis=0)
	#	for j, k in enumerate(knearest)]
	kdists = numpy.array([numpy.max([numpy.abs(b - chosen[j,:]) for b in chosen[k]], axis=0)
		for j, k in enumerate(knearest)])
	print(kdists[0].shape, kdists[0])
	plt.subplot(2, 3, i+1+3)
	plt.hist(clusterdists)
	
	plt.hist(kdists[:,0], alpha=0.3)
	plt.hist(kdists[:,1], alpha=0.3)
	plt.xlim(0, 1)
	maxdistance = 0.05
	maxdistance = numpy.max(kdists, axis=0)
	print('maxdistance:', maxdistance)
	ylims = plt.ylim()
	plt.vlines(maxdistance, ylims[0], ylims[1], linestyles=['-'])
	plt.ylim(ylims)
	
	plt.subplot(2, 3, i+1)
	# compute random points, check if they are included
	x = numpy.linspace(0, 1, 100)
	y = numpy.linspace(0, 1, 100)
	X, Y = numpy.meshgrid(x, y)
	sample = numpy.transpose([X, Y])
	#sample = numpy.random.uniform(0, 1, size=20*5000).reshape((-1, 2))
	# plot contour map of inclusion
	
	#dists = ((X.reshape((100,100,1)) - chosen[:,0].reshape((1,1,-1)))**2 + \
	#	 (Y.reshape((100,100,1)) - chosen[:,1].reshape((1,1,-1)))**2)**0.5
	
	dists = [numpy.abs(X.reshape(100, 100, 1) - chosen[:,0].reshape(1, 1, -1)),
		 numpy.abs(Y.reshape(100, 100, 1) - chosen[:,1].reshape(1, 1, -1)),
		]
	print(dists[0].shape)
	
	#dists = numpy.min(dists, axis=2)
	#dists = numpy.empty(len(sample))
	#for j, a in enumerate(chosen):
	#	dist = numpy.sum((a - sample)**2, axis=1)
	#	assert dist.shape == (len(sample),), dist.shape
	#	dists[j] = numpy.min(dist)
	closeby = numpy.any(numpy.all(dists < maxdistance.reshape((-1, 1, 1, 1)), axis=0), axis=2)
	print(closeby.shape)
	#plt.plot(X[-closeby], Y[-closeby], '.', color='grey', alpha=0.2)
	plt.contourf(X, Y, closeby*2., [-0.5, 0.5], colors=['grey'])
		
	
plt.savefig('friends.pdf', bbox_inches='tight')
plt.close()





