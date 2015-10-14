from __future__ import print_function
import scipy, scipy.stats
import numpy
from numpy import exp, log, log10
import numpy
import sklearn.svm
from nested_sampling.clustering import clusterdetect
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

def svm_classify(points, classes, plot=False):
	plot = plot and points.shape[1] == 2
	if plot:
		print('svm_classify plotting --')
		#plt.figure(figsize=(5,5))
		for c in sorted(set(classes)):
			x = points[:,0][classes == c]
			y = points[:,1][classes == c]
			plt.plot(x, y, 'x', ms=2)
			#print c, len(x), x[:10], y[:10]
		plt.savefig('svm_classifier.pdf', bbox_inches='tight')
		#plt.close()
		print('svm_classify plotting done.')
	u = points.mean(axis=0)
	s = points.std(axis=0)
	def transform(p):
		p = numpy.asarray(p)
		return (p - u) / s
	points = transform(points)
	clf = sklearn.svm.NuSVC(nu=0.05, probability=True, kernel='rbf')
	clf.fit(points, classes)
	#print clf.get_params()

	if plot:
		x = numpy.linspace(0, 1, 100)
		y = numpy.linspace(0, 1, 100)
		grid = numpy.array([[[xi, yi] for xi in x] for yi in y])
		dists = numpy.array([[clf.predict_proba(transform([[xi, yi]]))[0][0] for xi in x] for yi in y])
		print('levels:', dists.max(), dists.min())
		plt.contour(x, y, dists, [0.99, 0.9, 0.1], colors=['red', 'red', 'red'], linestyles=['-', '--', ':'])
		plt.savefig('svm_classifier_borders.pdf', bbox_inches='tight')
		plt.close()
	
	return clf, transform
	#lambda v: (clf.decision_function(v), clf.predict_proba(v)), clf

class SVMConstrainer(object):
	"""
	This constrainer uses probabilistic Support Vector Machines classifier
	with Radial basis functions (sklearn.svm.NuSVC)
	to find a border separating live points and already discarded points.
	
	Then, points are filtered using this classifier. Only points matching
	the classifier are evaluated.
	"""
	def __init__(self):
		self.sampler = None
		self.iter = 0
		self.rects = []
	
	def draw_constrained(self, Lmin, priortransform, loglikelihood, previous, ndim, **kwargs):
		# previous is [[u, x, L], ...]
		previousL = numpy.array([L for _, _, L in previous])
		previousu = numpy.array([u for u, _, _ in previous])
		assert previousu.shape[1] == ndim, previousu.shape
		self.iter += 1
		rebuild = self.iter % 50 == 1
		if rebuild:
			high = previousL > Lmin
			u = previousu[high]
			L = previousL[high]
		
			# detect clusters using hierarchical clustering
			assert len(u.shape) == 2, u.shape
			distances = scipy.spatial.distance.cdist(u, u)
			cluster = scipy.cluster.hierarchy.single(distances)
			
			n = len(distances)
			clusterdists = cluster[:,2]
			threshold = scipy.stats.mstats.mquantiles(clusterdists, 0.1)*20 + clusterdists.max()/2
			assigned = clusterdetect.cut_cluster(cluster, distances, threshold)
			# now we have clusters with some members
		
			# find some rough boundaries
			# make sure to make them so that they enclose all the points
			clusterids = sorted(set(assigned))
			rects = []
			for i in clusterids:
				inside = assigned == i
				ulow  = u[inside].min(axis=0)
				uhigh = u[inside].max(axis=0)
				width = uhigh - ulow
				# expand, to avoid over-shrinkage
				ulow  -= width*0.2
				uhigh += width*0.2
				j = L[inside].argmax()
				ustart = u[inside][j]
				Lstart = L[inside][j]
				rects.append((i, (ulow, uhigh, (log(uhigh - ulow)).sum())))
				#print 'adding new rectangle:', (i, (ulow, uhigh))
			rects = dict(rects)
			
			# now that we got a little more familiar with out clusters,
			# we want to sample from them
			# for this, we want to create boundaries between high and -high
			# we will do a multi-stage SVM, for every cluster
			rectid = numpy.zeros(len(previous), dtype=int) - 1
			rectid[high] = assigned
			#try:
			#if True:
			#if high.mean() >= 0.9:
			#	print 'not worth it yet to apply svm'
			clf, svmtransform = None, None
			if high.mean() < 0.9 and self.iter % 200 == 1:
				clf, svmtransform = svm_classify(previousu, rectid)
			#except ValueError as e:
			#	clf, svmtransform = None, None
			#	print 'WARNING: SVM step failed: ', e
			self.clf = clf
			self.svmtransform = svmtransform
			self.rects = rects
		
		if len(self.rects) == 1:
			def get_rect_id(x): return 0
		else:
			x = range(len(self.rects))
			y = numpy.array([self.rects[i][2] for i in x])
			minlogsize = y.min()
			y -= y.min()
			y = exp(y)
			totalsize = y.sum()
			y /= y.sum()
			y = [0] + y.cumsum().tolist() + [1]
			x = [x[0]] + list(x) + [x[-1]]
			get_rect_id = scipy.interpolate.interp1d(y, x, kind='zero')
		
		ntoaccept = 0
		while True:
			# sample from rectangles, and through against SVM
			dice = numpy.random.random()
			i = int(get_rect_id(dice))
			ulow, uhigh, logsize = self.rects[i]
			
			u = numpy.random.uniform(ulow, uhigh, size=ndim)
			if len(self.rects) != 1:
				# count in how many rectangles it is
				nrect = sum([exp(logsize - minlogsize)
					for ulow, uhigh, logsize in self.rects.values() if ((u >= ulow).all() and (u <= uhigh).all())])
				# reject proportionally  ~  1. / nrect
				coin = numpy.random.uniform(0, 1)
				accept = coin < exp(logsize) / nrect
				if not accept:
					continue
			
			# if survives (classified to be in high region)
			# then evaluate
			if self.clf is not None:
				prob = self.clf.predict_proba(self.svmtransform(u))[0][0]
				#print 'svm evaluation:', u, prob
				# we allow 1 false positive classified
				if prob > 1 - 1. / len(previous) and ntoaccept % 100 != 95:
					continue
			
			x = priortransform(u)
			L = loglikelihood(x)
			ntoaccept += 1
			if L >= Lmin:
				# yay, we win
				if ntoaccept > 5:
					print('%d samples before accept' % ntoaccept, u, x, L)
				return u, x, L, ntoaccept


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	#numpy.random.seed(0)
	points = numpy.random.uniform([0, 0], [1, 1], size=(1000, 2))
	#print 'points', points
	def priortransform(u):
		return u
	def loglikelihood(x):
		vals = 10*exp(-0.5 * (((x - numpy.array([0.4, 0.7]))/0.2)**2).sum())
		vals *= exp(-0.5 * ((((x[0] - 0.5)*2 - (x[1]*2)**3 + 0.01))/0.1)**2)
		vals += exp(-0.5 * (((x - numpy.array([0.2, 0.7]))/0.05)**2).sum())
		return log(vals)
	
	if True:
		vals = exp(-0.5 * (((points - numpy.array([0.4, 0.7]))/0.2)**2).sum(axis=1))
		vals *= exp(-0.5 * ((((points[:,0] - 0.5)*2 - (points[:,1]*2)**3 + 0.01))/0.1)**2)
		vals += exp(-0.5 * (((points - numpy.array([0.2, 0.7]))/0.05)**2).sum(axis=1))
		#print 'vals', vals
		high = vals > 0.01
		assert high.any()
		assert (-high).any()
		a = numpy.logical_and(points[:,0] < 0.4, high)
		b = numpy.logical_and(points[:,0] >= 0.4, high)
		plt.figure(figsize=(5,5))
		plt.plot(points[:,0][a],  points[:,1][a],  'x', color='b')
		plt.plot(points[:,0][b],  points[:,1][b],  'x', color='m')
		plt.plot(points[:,0][-high], points[:,1][-high], '+', color='g')
		plt.savefig('svmtest.pdf', bbox_inches='tight')
		clf, transform = svm_classify(points, high + a)
	
		x = numpy.linspace(0, 1, 100)
		y = numpy.linspace(0, 1, 100)
		#grid = numpy.array([[xi, yi] for xi in x for yi in y])
		grid = numpy.array([[[xi, yi] for xi in x] for yi in y])
		#X, Y = numpy.meshgrid(x, y)
		dists = numpy.array([[clf.predict_proba(transform([[xi, yi]]))[0][0] for xi in x] for yi in y])
		#print dists.shape
		#print grid.shape, dists.shape
		#print grid[:,0].shape, grid[:,1].shape, dists[:,0].shape
		#plt.contourf(grid[:,0], grid[:,1], dists[:,0], 5)
		plt.contourf(x, y, dists)
		plt.contour(x, y, dists, [0.99, 0.9, 0.1], colors=['red', 'red', 'red'], linestyles=['-', '--', ':'])
		plt.savefig('svmtest.pdf', bbox_inches='tight')
	
		points = numpy.random.uniform([0, 0], [1, 1], size=(1000, 2))
		results = clf.predict(transform(points))
		for (x, y), r in zip(points, results):
			#print x, y, r
			plt.plot(x, y, 'o', color='b' if r == 1 else 'g')
		plt.savefig('svmtest.pdf', bbox_inches='tight')
		plt.close()
	
	from simplenested import NestedSampler, nested_integrator
	constrainer = SVMConstrainer()
	
	print('preparing sampler')
	sampler = NestedSampler(nlive_points = 200, ndim=2,
		priortransform=priortransform, loglikelihood=loglikelihood, 
		draw_constrained = constrainer.draw_constrained)
	constrainer.sampler = sampler
	print('running sampler')
	result = nested_integrator(tolerance=0.01, sampler=sampler)

	try:
		x = numpy.array([x for _, x, _ in sampler.samples])
		y = numpy.exp([l for _, _, l in sampler.samples])
		print(x)
		print(y)
		plt.figure()
		plt.hexbin(x[:,0], x[:,1], C=y, gridsize=40)
		plt.savefig('svmtest_nested_samples.pdf', bbox_inches='tight')
		plt.close()
	except Exception as e:
		print(e)

	try:
		weights = numpy.array(result['weights']) # L, width
		plt.figure()
		plt.plot(exp(weights[:,1]), exp(weights[:,0]), 'x-', color='blue', ms=1)
		#plt.plot(weights[:,0], weights[:,1], 'x-', color='blue')
		plt.xlabel('prior mass')
		plt.ylabel('likelihood')
		#plt.xlim(0, 1)
		plt.savefig('svmtest_nested_integral.pdf', bbox_inches='tight')
		plt.close()
	except Exception as e:
		print(e)
	
	#u = numpy.linspace(0, 1, 10000)
	#x = numpy.array([priortransform(ui) for ui in u])
	#L = numpy.array([loglikelihood(xi) for xi in x])
	#print 'monte carlo integration (%d samples) logZ:' % len(u), log(exp(L).mean())

	print('nested sampling (%d samples) logZ = ' % len(result['samples']), result['logZ'], result['logZerr'])
	



