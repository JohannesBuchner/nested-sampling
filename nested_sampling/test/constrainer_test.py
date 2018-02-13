import json
import progressbar
from scipy.special import gamma, betainc, beta
import scipy.interpolate, scipy.stats
import numpy
from numpy import pi, fmod
import matplotlib.pyplot as plt

def distance(x, y):
	#diffs = numpy.array([fmod(x[i] - y[i] + 1.5, 1) - 0.5 for i in range(len(x))])
	#return (diffs**2).sum()**0.5
	diffs = x - y
	assert numpy.isfinite(diffs).all(), diffs
	return numpy.max(numpy.abs(diffs))
def nearest_neighbor(x, points):
	assert numpy.isfinite(x).all(), x
	assert numpy.isfinite(points).all(), points
	i, p = min(enumerate(points), key=lambda i_y: distance(x, i_y[1]))
	return i, p, distance(x, p)

def nearest_neighbor(x, points):
	distances = scipy.spatial.distance.cdist([x], points, metric='chebyshev')
	i = distances.argmin()
	return i, points[i], distances[0, i]

# cube of unit volume:
def vol(r, d):
	return r**d


def vol_cap(r, a):
	unit_volume = pi**(n/2.)/gamma(n/2.+1)
	V = unit_volume * r**n
	if a >= 0:
		return V / 2 * betainc((n+1)/2., 0.5, 1-(a/r)**2)
	else:
		return V - vol_cap(r, -a)

# n-sphere with radius r in origin
# intersecting with n-sphere at distance p2 with radius 1
# normalize volume so that at the largest radius, it is 1
def vol_intersect(r, d, dist):
	r2 = 1
	if dist > r + r2:
		# no intersection
		return
	elif dist < abs(r - r2):
		# one sphere is within the other
		# i.e. r can be at most
		# dist == r - r2 or dist == -(r - r2)
		# r = dist + r2  or r = r2 - dist
		# i.e. at most r can be dist + r2, then it encloses everything,
		# and the volume is related to r2**d
		return
	else:
		# two spherical caps.
		# compute triangle
		c1 = (dist**2 + r**2 - r2**2) / (2*dist)
		c2 = (dist**2 - r**2 + r2**2) / (2*dist)
		return vol_cap(r, c1) + vol_cap(r2, c2)
		

def prob(r, d, N):
	return (1 - vol(r, d))**(N-1) * N * vol(r, d) / r * d

def expect(d, N):
	return N * beta(1 + 1./d, N)

def plot_distribution(d, N, distances, shrinkages):
	plt.figure(figsize=(8, 5))
	nbins = 5
	green = '#ff6161'
	red = 'black'
	linestylea=dict(linestyle='dashdot', lw=1)
	linestyleb=dict(linestyle='--', lw=1)
	# shrinkage:
	shrinkages = numpy.array(shrinkages)
	x = numpy.linspace(0, min(2*shrinkages.max(), 1), 4000)
	rv = scipy.stats.beta(N, 1)
	pdf = rv.pdf
	cdf = rv.cdf
	mean = rv.mean()
	pdf = lambda x: (N * d) * (1 - x)**(d * N - 1)
	cdf = lambda x: 1 - (1 - x)**(d * N)
	mean = 1 - rv.mean()**(1. / d)
	
	p = pdf(x)
	c = cdf(x)
	plt.subplot(2, 2, 2)
	plt.locator_params(nbins=nbins)
	plt.plot(x[c<1], p[c<1], color=green, label='theoretical')

	plt.hist(shrinkages, normed=True, bins=40, histtype='step', color=red,
		label='observed')
	ymax = plt.ylim()[1]
	plt.legend(loc='upper right')
	plt.vlines(mean, 0, ymax, color=green, **linestylea)
	plt.vlines(shrinkages.mean(), 0, ymax, color=red, **linestyleb)
	#plt.ylabel('frequency')
	plt.ylim(0, ymax)
	#xmax = max(x[c<1][-1], shrinkages.max())
	xmax = shrinkages.max()
	#print 'maxx:', xmax
	plt.xlim(0, xmax)
	plt.locator_params(nbins=nbins)
	plt.subplot(2, 2, 4)
	plt.locator_params(nbins=nbins)
	plt.plot(x[c<1], cdf(x[c<1]), color=green)
	plt.hist(shrinkages, normed=True, cumulative=True, bins=4000, histtype='step', color=red)
	plt.vlines(mean, 0, 1, color=green, **linestylea)
	plt.vlines(shrinkages.mean(), 0, 1, color=red, **linestyleb)
	plt.ylim(0, 1)
	plt.xlim(0, xmax)
	
	# apply KS test
	D, pval = scipy.stats.kstest(shrinkages, cdf)
	plt.xlabel('shrinkage')
	#plt.ylabel('frequency')
	plt.text(xmax*0.45, 0.1, 'KS-test:\nDistance: %.4f\np-value: %.2f' % (D, pval),
		ha='left')

	plt.subplot(2, 2, 2)
	plt.locator_params(nbins=nbins)
	plt.title('Shrinkage test')
	#plt.title('KS-test Distance: %.4f, p-value: %.2f' % (D, pval))
	sD, spval = D, pval
	
	# 
	# neighbor distances:
	distances = numpy.array(distances)
	R = pi**-0.5 * gamma(d/2. + 1)**(1./d)
	R = 1
	#print 'R', R, vol(R, d)

	r = numpy.linspace(1e-20, R, 4000)

	p = prob(r, d=d, N=N)
	
	plt.subplot(2, 2, 1)
	plt.locator_params(nbins=nbins)
	# side length of enclosing rectangle, vs probability of being there
	plt.plot(r, p, color=green)

	# side length of enclosing rectangle, vs probability of being there
	plt.hist(2*distances, normed=True, bins=40, histtype='step', color=red)
	ymax = plt.ylim()[1]
	plt.vlines(expect(d=d, N=N), 0, ymax, color=green, **linestylea)
	plt.vlines((2*distances).mean(), 0, ymax, color=red, **linestyleb)
	#plt.ylabel('frequency')
	plt.ylim(0, ymax)
	plt.locator_params(nbins=nbins)
	plt.subplot(2, 2, 3)
	plt.locator_params(nbins=nbins)
	plt.plot(r, p.cumsum() / (4000 - 1), color=green)
	plt.hist(2*distances, normed=True, cumulative=True, bins=4000, histtype='step', color=red)
	plt.vlines(expect(d=d, N=N), 0, 1, color=green, **linestylea)
	plt.vlines((2*distances).mean(), 0, 1, color=red, **linestyleb)
	plt.ylim(0, 1)
	
	# apply KS test
	cdf = scipy.interpolate.interp1d(r, p.cumsum() / (4000 - 1), bounds_error=False, fill_value=0)
	D, pval = scipy.stats.kstest(2*distances, cdf)
	plt.xlabel('distance')
	plt.text(0.05, 0.1, 'KS-test:\nDistance: %.4f\np-value: %.2f' % (D, pval),
		ha='left')
	plt.ylabel('cumulative frequency')
	plt.locator_params(nbins=nbins)
	plt.subplot(2, 2, 1)
	plt.ylabel('frequency')
	plt.locator_params(nbins=nbins)
	#plt.title('KS-test Distance: %.4f, p-value: %.2f' % (D, pval))
	plt.title('Nearest Neighbour test')
	return dict(D=D, pvalue=pval, d=d, N=N,
		shrinkage_D=sD, shrinkage_pvalue=spval)

def generate_data(d, N):
	center = numpy.zeros(d) + 0.5
	distances = numpy.array([distance(center, nearest_neighbor(center, numpy.random.uniform(size=(N, d)))[1])
		for it in range(1000)])
	return distances

def normalized_distance(d, points, rsize, uj, verbose=False):
	# the prior constraint is at
	# 0.5 +- (points[i] - 0.5)
	# this region is of size:
	if verbose: print('rsize:', rsize)
	rvol = vol(r=rsize*2, d=d)
	if verbose: print('rvol:', rvol)
	scale = (rvol / 1) ** (1. / d)
	if verbose: print('scale:', scale)
	assert scale > 0, (scale, rvol, rsize, points)
	
	# move all points so that new point is in the center (0.5^d)
	center = uj
	points = numpy.array(points)
	assert points.shape[1] == len(center), (points.shape, center, points[0], len(center))
	
	#if verbose: print 'points:', points[:2]
	#assert numpy.isfinite(points).all(), points
	#shifted_points = [p - center for p in points]
	#if verbose: print 'shifted_points:', shifted_points[:2]
	## wrap around constraint border
	#wrapped_points = [fmod(p + 3*rsize, rsize) for p in shifted_points]
	#if verbose: print 'wrapped_points:', wrapped_points[:2]
	## scale all points so that the volume is 1
	#assert numpy.isfinite(wrapped_points).all(), wrapped_points
	#scaled_points = [p / scale for p in wrapped_points]
	#if verbose: print 'scaled_points:', scaled_points[:2]
	#assert numpy.isfinite(scaled_points).all(), scaled_points
	## find nearest neighbor point and
	scaled_points = numpy.fmod((points - center) + 3 * rsize, rsize) / scale
	# compute distance to it
	center = numpy.zeros(d) + 0.5
	
	l, nearest, dist = nearest_neighbor(center, scaled_points)
	if verbose: print('nearest to', center, ':', l, nearest)
	#dist = distance(center, nearest)
	if verbose: print('distance', dist)
	return dist

def shrinkage(d, rsize, rnewsize, verbose=False):
	""" Volume shrinkage factor """
	#Vtot = vol(r=rsize*2, d=d)
	#rnewsize = numpy.max([numpy.abs(p - 0.5).max() for p in points])
	
	ratio = (rnewsize / rsize)
	VolRatio = vol(r=ratio, d=d)
	assert numpy.isfinite(VolRatio), (rnewsize, rsize)
	#print rnewsize, rsize, VolRatio, (rsize - rnewsize) / rsize
	return (rsize - rnewsize) / rsize
	return VolRatio
	Vnew = vol(r=rnewsize*2, d=d)
	assert numpy.isfinite(Vtot), (Vtot, rsize)
	assert numpy.isfinite(Vnew), (Vnew, rsize)
	assert numpy.isfinite(Vnew / Vtot), (Vtot, Vnew, rsize, rnewsize)
	return Vnew / Vtot

def priortransform(x): return x
def loglikelihood(x):  return -numpy.abs(x-0.5).max()**0.01

def collect_from_multinest_constrainer(d, N, seed, niter=10000, verbose=False):
	print('collecting', d, N, seed)
	sequence = numpy.loadtxt('mn_pyramid_%d_%d_%d_sequence' % (d, N, seed))
	nested_samples = numpy.loadtxt('mn_pyramid_%d_%d_%d_.txt' % (d, N, seed))[:niter*10+2*N,:]
	# one has Lmin, L, n, the other has weight, -2*L
	assert sequence.shape[1] - 3 == nested_samples.shape[1] - 2
	#	dtype=[('Lmin', float), ('L', float), ('d', float), ('n', int)])
	#assert niter == 2000, niter
	# some points should be ignored, because of ellipse overlap
	#  -- specifically, those where the sample was doubled
	skip_sequence = []
	# select those coordinates/L that are in nested_samples
	coordinates_set = set([tuple(c) for c in nested_samples[:,2:]])
	coordinates = numpy.array(nested_samples[:,2:])
	
	# compute distances between coordinates and sequence[:,4:]
	#d = scipy.spatial.distance.cdist(coordinates, sequence[:,4:], metric='chebyshev')
	#print (d[:15000] < 1e-5).sum()
	
	#nprev = 0
	pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),
		progressbar.Counter('%5d'), progressbar.Bar(), progressbar.ETA()])
	for i, row in enumerate(pbar(sequence)):
		Lmin, L, n = row[:3]
		n = int(n)
		uj = row[3:]
		#nprev += n
		
		chosen = tuple(uj) in coordinates_set
		#if not chosen:
		#	scale = numpy.abs(uj - 0.5).max()
		#	dist = scipy.spatial.distance.cdist([(uj - 0.5) / scale], 
		#		(coordinates - 0.5) / scale, metric='chebyshev')
		#	chosen = (dist < 1e-5).any()
		#	if chosen:
		#		print (dist < 1e-5).sum(), dist
		#	mask = numpy.array([numpy.allclose(uj, c, rtol=1e-09, atol=1e-20) for c in coordinates])
		#	print mask.sum(), uj.shape, coordinates.shape
		#	chosen = mask.any()
		
		
		if chosen: # or numpy.allclose(uj, coordinates, rtol=1e-09, atol=1e-20):
			#row[3] = nprev
			skip_sequence.append(row)
			#nprev = 0
	
	#sequence = skip_sequence
	#skip_sequence = []
	#lastrow = sequence[0]
	#for i, row in enumerate(sequence[1:]):
	#	if lastrow[0] < -1e300 or row[0] != lastrow[0]:
	#		skip_sequence.append(lastrow)
	#	else:
	#		print 'double   %d:' % i, lastrow
	#		print 'for      %d:' % (i+1), row
	#		# add number of samples required
	#	lastrow = row
	#double   559: [-0.99252442 -0.99226279  0.45990741  3.          0.46586668  0.27797759
	#  0.95256335  0.04009259  0.73110932  0.10233873  0.37383509]
	#for      560: [-0.99252442 -0.9900788   0.36895728  1.          0.59133446  0.69940084
	#  0.17985809  0.73270625  0.6301567   0.13104272  0.28685457]
	
	print('sequence shortened from %d to %d (%.3f%%) from %d samples' % (len(sequence), len(skip_sequence),
		len(skip_sequence) * 100. / len(sequence), len(coordinates)))
	assert len(coordinates) == len(skip_sequence)
	assert niter + 2*N <= len(skip_sequence)
	live_points = {}
	total_samples = 0
	
	distances = []
	shrinkages = []
	# go through posterior chain of MultiNest. consider likelihood values
	# skip those in between, keep adding up n
	i = 0
	pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),
		progressbar.Counter('%5d'), progressbar.Bar(), progressbar.ETA()])
	for row in pbar(skip_sequence[:N+niter]):
		Lmin, L, n = row[:3]
		n = int(n)
		uj = row[3:]
		
		# initial fill-up
		if Lmin < -1e300:
			assert L not in live_points, L
			live_points[L] = [uj]
			continue
		total_samples += n
		
		#rsize    = (-Lmin)**100
		#rnewsize = (-L)**100
		#shrinkages.append(shrinkage(d, rsize, rnewsize, verbose=verbose))
		
		# previous point
		assert Lmin in live_points, Lmin
		ui = live_points[Lmin][-1]
		rsize = numpy.abs(ui - 0.5).max()
		#rsize = 0.5 - (-Lmin)**100
		# all live points
		points = []
		for p in live_points.values():
			points += p
		assert len(points) == N
		if verbose and i < 40: print('row:', len(points), Lmin, L, rsize, n, uj)
		
		dist = normalized_distance(d, points, rsize, uj, verbose=verbose)
		if verbose and i < 40: print('distance:', dist, rsize, -((0.5 - rsize)**0.01))
		# store distance
		distances.append(dist)
		
		# replace point
		live_points[Lmin].pop()
		if not live_points[Lmin]:
			del live_points[Lmin]
		live_points[L] = live_points.get(L, []) + [uj]
		
		# store shrinkage: use points after removing least likely point
		points = []
		for p in live_points.values():
			points += p
		rnewsize = numpy.abs(numpy.array(points) - 0.5).max()
		shrinkages.append(shrinkage(d, rsize, rnewsize, verbose=verbose))
		
		i = i + 1
		if i > niter:
			break
	return distances, shrinkages, total_samples, niter

def evaluate_multinest_constrainer(d, N, niter=10000, verbose=False):
	distances = []
	shrinkages = []
	total_samples = 0
	niters = 0
	for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
		dists, s, tot, n = collect_from_multinest_constrainer(d, N, seed, niter=niter, verbose=verbose)
		distances += dists
		shrinkages += s
		total_samples += tot
		niters += n

	# plot distance distribution
	# vs prediction
	results = plot_distribution(d, N, distances, shrinkages)
	results['total_samples'] = total_samples
	results['niter'] = niters
	return results

#if __name__ == '__main__':
#	evaluate_multinest_constrainer(d=2, N=400, niter=10000, verbose=False)
#	import sys; sys.exit(0)


def sample_from_constrainer(d, N, constrainer, seed, niter=10000, verbose=False):
	numpy.random.seed(seed)
	# use 400 points
	points = list(numpy.random.uniform(size=(N, d)))
	values = [loglikelihood(x) for x in points]
	if verbose: print('points:', list(zip(points, values)))
	
	distances = []
	shrinkages = []
	previous = list(zip(points, points, values))
	
	pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(),
		progressbar.Counter('%5d'), progressbar.Bar(), progressbar.ETA()])
	total_samples = 0
	for it in pbar(range(niter)):
		# remove lowest, draw a higher one
		i = numpy.argmin(values)
		k = numpy.random.randint(0, N-1)
		if k >= i:
			k += 1
		Li = values[i]
		ui = points[i]
		xi = points[i]
		# reached numerical accuracy: all points are in the center
		if numpy.all(ui == 0.5):
			niter = it
			break
		assert numpy.isfinite(Li), Li
		assert numpy.isfinite(ui).all(), ui
		if verbose: print('calling draw_constrained with Lmin', Li, ui)

		uj, xj, Lj, n = constrainer.draw_constrained(
			Lmin=Li, 
			priortransform=priortransform, 
			loglikelihood=loglikelihood, 
			previous=previous,
			ndim=d,
			draw_global_uniform=lambda: numpy.random.uniform(0, 1, size=d),
			startu = points[k], 
			startx = points[k], 
			startL = values[k],
			starti = i)
		assert numpy.isfinite(uj).all(), uj
		assert numpy.isfinite(Lj), Lj
		total_samples += n
		
		rsize = numpy.abs(ui - 0.5).max()
		dist = normalized_distance(d, points, rsize, uj, verbose=verbose)
		
		# store distance
		distances.append(dist)
		
		# replace point
		points[i] = uj
		values[i] = Lj
		previous.append([uj, uj, Lj])

		# store shrinkage: use points after removing least likely point
		rnewsize = numpy.abs(numpy.array(points) - 0.5).max()
		shrinkages.append(shrinkage(d, rsize, rnewsize, verbose=verbose))
	return distances, shrinkages, total_samples, niter


def evaluate_constrainer(d, N, constrainer, niter=10000, verbose=False):
	distances = []
	shrinkages = []
	total_samples = 0
	niters = 0
	for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
		dists, s, tot, n = sample_from_constrainer(d, N, constrainer(), seed, niter=niter, verbose=verbose)
		distances += dists
		shrinkages += s
		total_samples += tot
		niters += n

	# plot distance distribution
	# vs prediction
	results = plot_distribution(d, N, distances, shrinkages)
	results['total_samples'] = total_samples
	results['niter'] = niters
	return results
	
def test_volume():
	N = 400
	for d in 2, 7, 20:
		# generate uniform data
		distances = generate_data(d, N)
		plot_distribution(d, N, distances)
		plt.savefig('test_volumes_%d.pdf' % d, bbox_inches='tight')
		plt.close()

def run_constrainer(d, N, constrainer, name):
	filename = 'constrainer_results_%d_%s.json' % (d, name)
	try:
		results = json.load(open(filename))
	except IOError:
		# the rejection constrainer is really inefficient. stop early
		niter = 10000
		if d > 15:
			if name == 'rejection':
				niter = 4000
			else:
				niter = 4000
		elif name == 'rejection':
			niter = 4000
		
		print('running', name)
		if name == 'multinest':
			results = evaluate_multinest_constrainer(d=d, N=N, niter=niter)
		else:
			results = evaluate_constrainer(d=d, 
				N=N, constrainer=constrainer, 
				niter=niter)
		plt.savefig('test_constrainer_%d_%s.pdf' % (d, name), bbox_inches='tight')
		plt.close()
		results.update(dict(name=name, efficiency=results['niter'] * 100. / results['total_samples']))
		json.dump(results, open(filename, 'w'), indent=4)
	print('%(name)30s:  %(d)3d  %(N)3d  %(D).2f  %(pvalue).4f  %(shrinkage_D).2f  %(shrinkage_pvalue).4f %(niter)6d  %(total_samples)10d  %(efficiency).2f%%' % results)
	return results


