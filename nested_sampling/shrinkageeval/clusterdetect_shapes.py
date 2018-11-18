from clustering import clusterdetect 
import matplotlib.pyplot as plt
import numpy
from numpy import log10, log, exp
import scipy.spatial, scipy.cluster, scipy.stats
import itertools
#numpy.random.seed(1)

params = []
nresample = 5
for iresample in range(nresample):
	single = numpy.random.normal(0, 0.1, size=200).reshape((100, 2)) + [0.1, 0.7]

	single2 = numpy.random.normal(0, 0.04, size=100).reshape((50, 2)) + [0.5, 0.2]
	double = numpy.vstack((single, single2))

	coupled = [numpy.vstack((single, single + [d, -d])) for d in [0.3, 0.5, 0.6, 5]]

	sigma = 0.01
	r = 0.93 * sigma
	along = numpy.random.multivariate_normal([0.3, 0.7], [[sigma, r], [r, sigma]], size=100)


	along2 = numpy.vstack((along, along + [0.2, -0.2]))

	samples = [single, double, along, along2] + coupled

	plt.figure(figsize=(2+len(samples)*2, 6))
	for j, data in enumerate(samples):
		colors = itertools.cycle('r,g,b,m,k,y,orange,grey'.split(','))
	
		plt.subplot(3, len(samples), 1+j)
		plt.plot(data[:,0], data[:,1], 'x')
		plt.xlim(0, 1)
		plt.ylim(0, 1)
		
		u = data
		# detect clusters using hierarchical clustering
		distances = scipy.spatial.distance.cdist(u, u)
		cluster = scipy.cluster.hierarchy.single(distances)

		n = len(distances)
		clusterdists = cluster[:,2]
		# the idea here is that the inter-cluster distances must be much smaller than the cluster-distances
		# small cluster distances multiplied by 20 will remain small, if there is a well-separated cluster
		# if no cluster, then this will be ~clusterdists.max()/2, and no splitting will be done
		threshold = scipy.stats.mstats.mquantiles(clusterdists, 0.1)*10 + clusterdists.max()/3
		
		plt.subplot(3, len(samples), len(samples)+1+j)
		plt.hist(clusterdists, histtype='step', cumulative=True, normed=True, bins=10000)
		plt.yticks([])
		plt.gca().set_xscale('log')
		ylim = plt.ylim()
		cdf = numpy.copy(clusterdists)
		cdf.sort()
		# fit gaussian
		def minfunc(gaussparams):
			# ks statistic, trimmed
			(mu, sigma) = gaussparams
			n = len(clusterdists)
			p = numpy.linspace(0, 1, n)
			cdf = scipy.stats.norm(mu, sigma).cdf(log10(clusterdists))
			#print mu, sigma, cdf, p
			return numpy.max( numpy.abs(cdf - p)[n*2/10 : n*9/10])
			#d, p = scipy.stats.kstest(log10(clusterdists), rv)
			#return d
		
		mu, sigma = log10(clusterdists).mean(), log10(clusterdists).std()
		rv = scipy.stats.norm(mu, sigma)
		x = numpy.linspace(0, 1, 1000)
		plt.plot(10**rv.ppf(x), x, '-', color='r', alpha=0.1)
		mu, sigma = scipy.optimize.fmin(minfunc, [mu, sigma], disp=0)
		rv = scipy.stats.norm(mu, sigma)
		plt.plot(10**rv.ppf(x), x, '-', color='r')
		
		#   max * (1/2 + q10 / max * 20)
		# if it were a unimodal distribution, 3 sigma would be rare
		# what fraction are at low and high bounds
		D, p = scipy.stats.kstest(clusterdists, rv.cdf)
		params.append([D, p])
		print(D, p)
		plt.ylabel('1 - D = %f' % (1 - D))
		if D > 0.997 and False:
			assigned = numpy.zeros(len(u))
		else:
			assigned = clusterdetect.cut_cluster(cluster, distances, threshold)
		# now we have clusters with some members
		
		# find some rough boundaries
		# make sure to make them so that they enclose all the points
		clusterids = sorted(set(assigned))
	
	
		plt.vlines(scipy.stats.mstats.mquantiles(clusterdists, [0.1, 0.75]), ylim[0], ylim[1], linestyle='-', colors=['g', 'b'])
		plt.vlines(clusterdists.max()/2, ylim[0], ylim[1], linestyle='-', color='k')
		plt.vlines(threshold, ylim[0], ylim[1], linestyle='-', color='r')
		plt.ylim(0, 1)

		plt.subplot(3, len(samples), len(samples)*2+1+j)
		for i in clusterids:
			inside = assigned == i
			chosen = u[inside]
			plt.plot(chosen[:,0], chosen[:,1], 'x', color=next(colors))
		plt.xlim(0, 1)
		plt.ylim(0, 1)

	plt.savefig('x_shapes_.pdf', bbox_inches='tight')
	plt.close()

plt.figure()
params = numpy.array(params)

colors = itertools.cycle('r,g,b,m,k,y,orange,grey'.split(','))
for i in range(nresample):
	plt.plot(params[i::nresample,0], params[i::nresample,1], 'x', color=next(colors))
plt.savefig('x_shapes_norm.pdf', bbox_inches='tight')
plt.close()


