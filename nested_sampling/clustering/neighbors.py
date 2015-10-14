from __future__ import print_function
import numpy
import scipy.spatial

def initial_maxdistance_guess(u):
	n = len(u)
	distances = scipy.spatial.distance.cdist(u, u)
	nearest = [distances[i,:].argsort()[1] for i in range(n)]
	nearest = [numpy.abs(u[k,:] - u[i,:]) for i, k in enumerate(nearest)]
	# compute distance maximum
	maxdistance = numpy.max(nearest, axis=0)
	return maxdistance

def update_maxdistance(u, ibootstrap, maxdistance, verbose = False):
	n, ndim = u.shape
	
	# bootstrap to find smallest maxdistance which includes
	# all points
	choice = list(set(numpy.random.choice(numpy.arange(n), size=n)))
	notchosen = set(range(n)) - set(choice)
	# check if included with our starting criterion
	for i in notchosen:
		dists = numpy.abs(u[i,:] - u[choice,:])
		close = numpy.all(dists < maxdistance.reshape((1,-1)), axis=1)
		assert close.shape == (len(choice),), (close.shape, len(choice))
		# find the point where we have to increase the least
		if not close.any():
			# compute maxdists -- we already did that
			# compute extension to maxdistance
			#maxdistance_suggest = [numpy.max([maxdistance, d], axis=0) for d in dists]
			maxdistance_suggest = numpy.where(maxdistance > dists, dists, maxdistance)
			assert maxdistance_suggest.shape == (len(dists), ndim)
			# compute volume increase in comparison to maxdistance
			#increase = [(numpy.log(m) - numpy.log(maxdistance)).sum()  for m in maxdistance_suggest]
			increase = numpy.log(maxdistance_suggest).sum(axis=1) - numpy.log(maxdistance).sum()
			
			# choose smallest
			nearest = numpy.argmin(increase)
			if verbose: print(ibootstrap, 'nearest:', u[i], u[nearest], increase[nearest])
			# update maxdistance
			maxdistance = numpy.where(dists[nearest] > maxdistance, dists[nearest], maxdistance)
			if verbose: print(ibootstrap, 'extending:', maxdistance)
		else:
			# we got this one, everything is fine
			pass
	return maxdistance

def find_maxdistance(u, verbose=False, nbootstraps=15):
	# find nearest point for every point
	if verbose: print('finding nearest neighbors:')
	maxdistance = initial_maxdistance_guess(u)
	#maxdistance = numpy.zeros(ndim)
	if verbose: print('initial:', maxdistance)
	for ibootstrap in range(nbootstraps):
		maxdistance = update_maxdistance(u, ibootstrap, maxdistance, verbose=verbose)
	return maxdistance


def nearest_rdistance_guess(u, metric='euclidean'):
	n = len(u)
	distances = scipy.spatial.distance.cdist(u, u, metric=metric)
	numpy.fill_diagonal(distances, 1e300)
	nearest_neighbor_distance = numpy.min(distances, axis = 1)
	rdistance = numpy.max(nearest_neighbor_distance)
	#print 'distance to nearest:', rdistance, nearest_neighbor_distance
	return rdistance
def initial_rdistance_guess(u, metric='euclidean', k = 10):
	n = len(u)
	distances = scipy.spatial.distance.cdist(u, u, metric=metric)
	#if k == 1:
	#	numpy.diag(distances)
	#	nearest = [distances[i,:])[1:k] for i in range(n)]
	#else:
	nearest = [numpy.sort(distances[i,:])[1:k+1] for i in range(n)]
	# compute distance maximum
	rdistance = numpy.max(nearest)
	return rdistance

def update_rdistance(u, ibootstrap, rdistance, verbose = False, metric='euclidean'):
	n, ndim = u.shape
	
	# bootstrap to find smallest rdistance which includes
	# all points
	choice = set(numpy.random.choice(numpy.arange(n), size=n))
	mask = numpy.array([c in choice for c in numpy.arange(n)])
	
	distances = scipy.spatial.distance.cdist(u[mask], u[-mask], metric=metric)
	assert distances.shape == (mask.sum(), (-mask).sum())
	nearest_distance_to_members = distances.min(axis=0)
	newrdistance = max(rdistance, nearest_distance_to_members.max())
	if newrdistance > rdistance and verbose:
		print( ibootstrap, 'extending:', rdistance)
	return newrdistance

def find_rdistance(u, verbose=False, nbootstraps=15, metric='euclidean'):
	# find nearest point for every point
	if verbose: print('finding nearest neighbors:')
	rdistance = 0 #initial_rdistance_guess(u)
	if verbose: print('initial:', rdistance)
	for ibootstrap in range(nbootstraps):
		rdistance = update_rdistance(u, ibootstrap, rdistance, verbose=verbose, metric=metric)
	return rdistance


