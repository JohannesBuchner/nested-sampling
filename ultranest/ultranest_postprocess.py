from __future__ import print_function
import numpy
from numpy import exp, log, log10
from operator import getitem, attrgetter
from nested_sampling.clustering import clusterdetect
import scipy.stats, scipy.cluster
import matplotlib.pyplot as plt

"""
Postprocessing the nested sampling results:

* creates equal weighted posterior samples from the weighted posterior samples
* detects modes and evaluates their relative importance (local evidence)
"""

def equal_weighted_posterior(weights, size=10000):
	"""
	Resamples the weighted posterior samples to get posterior samples with
	weights 1.
	
	weights is an array of entries [u, x, L, logwidth]
	where u are the untransformed parameters, x the transformed parameters,
	L the log-Likelihood and logwidth the volume the point represents.
	
	Returns: an array of u, and an array of x, each of length size.
	"""
	
	u = [ui for ui, xi, Li, logwidth in weights]
	x = [xi for ui, xi, Li, logwidth in weights]
	probs = numpy.array([Li + logwidth for ui, xi, Li, logwidth in weights])
	# avoid large numbers
	probs = exp(probs - probs.max())
	# numpy.random.choice needs normalization
	probs /= probs.sum()
	
	indices = numpy.arange(len(weights))
	# draw randomly
	choices = numpy.random.choice(indices, replace=True, p=probs, size=size)
	return numpy.array([u[i] for i in indices]), numpy.array([x[i] for i in indices])

def marginal_plot(x, w, i, j = None, grid_points = 40, **kwargs):
	if j is None:
		hist, bins = numpy.histogram(x[:,i], weights=w, bins=100)
		hist /= hist.max()
		plt.bar((bins[:-1] + bins[1:])/2., hist, width=(bins[1:] - bins[:-1]),
			color='grey', alpha=0.3, linewidth=0)
		plt.hist(x[:,i], weights=w, histtype='step', color='blue', 
			normed=True, bins=1000, cumulative=True)
		plt.ylim(0, 1)
	else:
		plt.hexbin(x=x[:,i], y=x[:,j], gridsize=grid_points, 
			C = w, reduce_C_function=numpy.nansum,
                        **kwargs)

def marginal_plots(weights, ndim):
	x = numpy.array([xi for ui, xi, Li, logwidth in weights])
	logw = numpy.array([Li + logwidth for ui, xi, Li, logwidth in weights])
	w = numpy.exp(logw - logw.max())
	for i in range(ndim):
		plt.subplot(ndim, ndim, i + 1)
		marginal_plot(x, w, i=i)
		for j in range(i):
			plt.subplot(ndim, ndim, i + (j+1)*ndim + 1)
			marginal_plot(x, w, i=i, j=j,
				cmap = plt.cm.gray_r)

def marginal_plots1d(weights, ndim):
	x = numpy.array([xi for ui, xi, Li, logwidth in weights])
	logw = numpy.array([Li + logwidth for ui, xi, Li, logwidth in weights])
	w = numpy.exp(logw - logw.max())
	for i in range(ndim):
		plt.subplot(ndim, 1, i + 1)
		marginal_plot(x, w, i=i)
	

def _weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    # https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    average = numpy.average(values, weights=weights)
    variance = numpy.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def mode_importance(weights):
	"""
	Detects separate modes in the posterior using clustering of the top 99% of the posterior.
	
	weights is an array of entries [u, x, L, logwidth]

	Returns: A list of modes with their median, standard deviation, and 
	local evidence.
	"""
	
	# strip lowest 1% of posterior
	weights = sorted(weights, key=getitem(2))
	u, x, L, lw = weights[-1]
	# use highest as normalization to avoid small numbers
	offset = L + lw
	total = sum([exp(L + lw - offset) for u, x, L, lw in weights])
	logtotal = log(total) + offset
	
	bottom_sum = 0
	high_parts = []
	for i, (u, x, L, lw) in enumerate(weights):
		bottom_sum += exp(L + lw)
		if bottom_sum > total * 0.01:
			high_parts.append(i)
	
	# perform clustering on the top
	# in prior space or in transformed space?
	# transformed space is probably pretty weird
	
	pos = [u for u, x, L, lw in weights[high_parts]]

	distances = scipy.spatial.distance.cdist(pos, pos)
	cluster = scipy.cluster.hierarchy.single(distances)

	# the idea here is that the inter-cluster distances must be much smaller than the cluster-distances
	# small cluster distances multiplied by 10 will remain small, if there is a well-separated cluster
	# if no cluster, then this will be ~clusterdists.max()/3, and no splitting will be done
	threshold = scipy.stats.mstats.mquantiles(clusterdists, 0.1)*10 + clusterdists.max()/3

	assigned = clusterdetect.cut_cluster(cluster, distances, threshold)
	# now we have clusters with some members
	
	clusterids = sorted(set(assigned))
	results = []
	for i in clusterids:
		inside = assigned == i
		inweights = weights[high_parts][inside]
		membersu = [x for u, x, L, lw in inweights]
		membersx = [u for u, x, L, lw in inweights]
		membersw = [L + lw for u, x, L, lw in inweights]
		probs = exp(numpy.array(membersw) - offset)
		# compute weighted mean 
		# compute weighted standard deviation
		umean, ustdev = _weighted_avg_and_std(values=membersu, weights=probs)
		xmean, xstdev = _weighted_avg_and_std(values=membersx, weights=probs)
		# compute evidence
		local_evidence = log(sum(probs)) + offset
		relative_evidence = exp(log(sum(probs)) - log(total))
		results.append(dict(
			members = [membersu, membersx, membersw],
			mean = xmean,
			stdev = xstdev,
			untransformed_mean = umean,
			untransformed_stdev = ustdev,
			relative_posterior_probability = relative_evidence,
			local_evidence = local_evidence,
		))
	return results
	
if __name__ == '__main__':
	import sys
	prefix = sys.argv[1]
	data = numpy.loadtxt(prefix)
	params = data[:,:-2]
	nparams = params.shape[1]
	L = data[:,-2]
	weight = data[:,-1]
	weights = list(zip(params, params, L, weight))
	
	print('creating posterior samples...')
	_, post_samples = equal_weighted_posterior(weights)
	numpy.savetxt(prefix + 'posterior_samples.txt', post_samples)
	
	print('creating marginal plots...')
	#if nparams > 10:
	plt.figure(figsize=(2+nparams, 2+nparams))
	marginal_plots(weights, nparams)
	plt.savefig(prefix + 'marginals.pdf', bbox_inches='tight')
	plt.close()
	
	modes = mode_importance(weights)
	for mode in modes:
		print()
		print('Mode')
		print('  Mean:', mode['mean'])
		print('  Stdev:', mode['stdev'])
		print('  importance:', mode['relative_posterior_probability'])
	









