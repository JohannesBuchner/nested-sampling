import scipy, scipy.stats, scipy.cluster, scipy.optimize
import numpy
from numpy import exp, log, log10
import numpy
import sklearn.svm
from nested_sampling.clustering import clusterdetect
import matplotlib.pyplot as plt

def svm_classify(points, classes, plot=True):
	plot = plot and points.shape[1] > 1
	if plot:
		#print 'svm_classify plotting --'
		plt.figure(figsize=(5,5))
		for c in sorted(set(classes)):
			x = points[:,0][classes == c]
			y = points[:,1][classes == c]
			plt.plot(x, y, 'x', ms=2)
			#print c, len(x), x[:10], y[:10]
		#plt.savefig('svm_classifier.pdf', bbox_inches='tight')
		#print 'svm_classify --'
	u = points.mean(axis=0)
	s = points.std(axis=0)
	transform = lambda p: (p - u)/s
	points = transform(points)
	clf = sklearn.svm.NuSVC(nu=0.05, probability=True, kernel='rbf')
	clf.fit(points, classes)

	if plot:
		x = numpy.linspace(0, 1, 100)
		y = numpy.linspace(0, 1, 100)
		grid = numpy.array([[[xi, yi] for xi in x] for yi in y])
		dists = numpy.array([[clf.predict_proba(transform([[xi, yi]]))[0][0] for xi in x] for yi in y])
		print 'levels:', dists.max(), dists.min()
		plt.contour(x, y, dists, [0.99, 0.9, 0.1], colors=['red', 'red', 'red'], linestyles=['-', '--', ':'])
		plt.savefig('svm_classifier.pdf', bbox_inches='tight')
		plt.close()
	
	return clf, transform
	#lambda v: (clf.decision_function(v), clf.predict_proba(v)), clf

class SVMConstrainer(object):
	def __init__(self, optimizer = scipy.optimize.fmin):
		self.maxima = []
		self.optimizer = optimizer
		self.sampler = None
		self.iter = 0
	
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
				j = L[inside].argmax()
				ustart = u[inside][j]
				Lstart = L[inside][j]
				assert len(ulow) == ndim
				assert len(uhigh) == ndim
				assert len(ustart) == ndim
				
				# find maximum in each cluster
				isinside = lambda ui: (ui >= ulow).all() and (ui <= uhigh).all()
				assert isinside(ustart)
				clustermaxima = [[mu, mL] for mu, mL in self.maxima if isinside(mu)]
				if len(clustermaxima) == 0:
					print 'optimizing in cluster', i, ulow, uhigh
					def minfunc(ui):
						if not isinside(ui):
							return 1e300
						return -loglikelihood(priortransform(ui))
					ubest = self.optimizer(minfunc, ustart)
					assert len(ubest) == ndim
					#ulow = numpy.min([ulow, ubest], axis=0)
					#uhigh = numpy.max([uhigh, ubest], axis=0)
					Lbest = loglikelihood(priortransform(ubest))
					print 'new best:', ubest, Lbest
					if self.sampler:
						self.sampler.Lmax = max(self.sampler.Lmax, Lbest)
					self.maxima.append([ubest, Lbest])
				else:
					if len(clustermaxima) > 1:
						print 'WARNING: multiple maxima fitted already', clustermaxima
					ubest, Lbest = clustermaxima[0]
				
				rects.append((i, (ulow, uhigh, ubest, Lbest)))
				print 'adding new rectangle:', (i, (ulow, uhigh, ubest, Lbest))
			rects = dict(rects)
		
			# now that we got a little more familiar with out clusters,
			# we want to sample from them
			# for this, we want to create boundaries between high and -high
			# we will do a multi-stage SVM, for every cluster
			rectid = numpy.zeros(len(previous), dtype=int) - 1
			rectid[high] = assigned
			try:
				if high.mean() >= 0.9:
					raise ValueError('not worth it yet')
				clf, svmtransform = svm_classify(previousu, rectid)
			except ValueError as e:
				clf, svmtransform = None, None
				print 'WARNING: SVM step failed: ', e
			self.clf = clf
			self.svmtransform = svmtransform
			self.rects = rects
		
		ntoaccept = 0
		while True:
			# sample from rectangles, and through against SVM
			i = numpy.random.randint(0, len(self.rects))
			ulow, uhigh, ubest, Lbest = self.rects[i]
			
			assert len(ulow) == ndim
			assert len(uhigh) == ndim
			u = numpy.random.uniform(ulow, uhigh, size=ndim)
			assert len(u) == ndim
			
			# count in how many rectangles it is
			nrect = sum([((u >= ulow).all() and (u <= uhigh).all()) for ulow, uhigh, ubest, Lbest in self.rects.values()])
			
			# reject proportionally
			if nrect > 1 and numpy.random.uniform(0, 1) > 1./nrect:
				continue
			
			# if survives (classified to be in high region)
			# then evaluate
			if self.clf is not None:
				prob = self.clf.predict_proba(self.svmtransform(u))[0][0]
				#print 'svm evaluation:', u, prob
				if prob > 1 - 1./len(previous) and ntoaccept % 100 != 95:
					continue
			
			x = priortransform(u)
			L = loglikelihood(x)
			ntoaccept += 1
			if L > Lmin:
				# yay, we win
				#print '%d samples before accept' % ntoaccept, u, x, L
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
	
	print 'preparing sampler'
	sampler = NestedSampler(nlive_points = 200, ndim=2,
		priortransform=priortransform, loglikelihood=loglikelihood, 
		draw_constrained = constrainer.draw_constrained)
	constrainer.sampler = sampler
	print 'running sampler'
	result = nested_integrator(tolerance=0.01, sampler=sampler)

	try:
		x = numpy.array([x for _, x, _ in sampler.samples])
		y = numpy.exp([l for _, _, l in sampler.samples])
		print x
		print y
		plt.figure()
		plt.hexbin(x[:,0], x[:,1], C=y, gridsize=40)
		plt.savefig('svmtest_nested_samples.pdf', bbox_inches='tight')
		plt.close()
	except Exception as e:
		print e

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
		print e
	
	#u = numpy.linspace(0, 1, 10000)
	#x = numpy.array([priortransform(ui) for ui in u])
	#L = numpy.array([loglikelihood(xi) for xi in x])
	#print 'monte carlo integration (%d samples) logZ:' % len(u), log(exp(L).mean())

	print 'nested sampling (%d samples) logZ = ' % len(result['samples']), result['logZ'], result['logZerr']
	



