from clustering.clusterdetect import *
import matplotlib.pyplot as plt

def loglikelihood(x):
	a = - 0.5 * ((x - 0.2)/0.05)**2
	b = - 0.5 * ((x - 0.7)/0.05)**2
	return exp(a) + exp(b) + 0.01

numpy.random.seed(0)
values = numpy.random.uniform(0, 1, size=50)
samples = loglikelihood(values)

plt.plot(values, samples, 'x')
plt.savefig('x_points.pdf', bbox_inches='tight')
plt.close()
sortx = numpy.argsort(values)
vals = samples[sortx].cumsum() 
#plt.plot(values[sortx], numpy.linspace(0, 1, len(values)), 'x-', label='likelihood cdf')
#plt.plot(values[sortx], vals / vals.max(), 'x-', label='likelihood cdf')
weights = values[sortx][1:] - values[sortx][:-1]
vals = (numpy.min([samples[sortx][:-1], samples[sortx][1:]], axis=0) * weights).cumsum()
plt.plot(values[sortx][1:], vals / vals[-1], 'x-', label='min step function')
vals = (numpy.max([samples[sortx][:-1], samples[sortx][1:]], axis=0) * weights).cumsum()
plt.plot(values[sortx][:-1], vals / vals[-1], 'x-', label='max step function')
vals = ((samples[sortx][:-1] + samples[sortx][1:])/2. * weights).cumsum()
plt.plot(values[sortx][:-1], vals / vals[-1], 'x-', label='trapez function')
vals = samples[sortx].cumsum()
plt.plot(values[sortx], vals / vals[-1], 'x-', label='point function')
plt.plot([0, 1], [0, 1], '--', color='grey')
plt.legend(loc='upper left')
plt.savefig('x_cdf.pdf', bbox_inches='tight')
plt.close()

# find humps
# or find lack of humps
##lack of humps = cdf goes below uniform prediction
#mask = vals / vals[-1] > numpy.linspace(0, 1, len(vals))
## by likelihood
#mask = samples[sortx] > samples[sortx].max() / 100
## by posterior mass contribution
# select those where cdf is pretty flat
vals = (numpy.max([samples[sortx][:-1], samples[sortx][1:]], axis=0) * weights)
vals /= vals.sum()
mask = vals < 0.01 / len(samples)
print 'small contributions:', mask.sum()
# select maximum likelihood of those
maxlike = samples[sortx][mask].max()
print 'max like in small contributions:', maxlike
mask = samples[sortx] > maxlike

# use points above this criterion
assert mask.sum() > 3
print 'selecting %d of %d for clustering' % (mask.sum(), len(mask))
highpoints = numpy.reshape(values[sortx][mask], (-1, 1))
print 'distance matrix...'
distances = scipy.spatial.distance.cdist(highpoints, highpoints)

# make dendrogram based on euclidean distances, always merge smallest distance
# make histogram of distances
print 'clustering...'
cluster = scipy.cluster.hierarchy.single(distances)

assert len(cluster) == len(distances) - 1
n = len(cluster) + 1
tree = {}
fout = open('x_clustergraph.dot', 'w')
fout.write("digraph g{\n")
for i, (a, b, dist, entries) in enumerate(cluster):
	#print '%3d%s %3d%s %.2f %3d' % (a, 
	#	'*' if a < n else ' ', b, '*' if b < n else ' ', dist, entries)
	fout.write("%d -> %d [label=%.2f];\n" % (i+n, a, dist))
	fout.write("%d -> %d [label=%.2f];\n" % (i+n, b, dist))
	tree[i + n] = (a, b, dist, entries)
fout.write("}\n")
# from cluster index 0 and 1 are combined in row i to form index n + i
#     if index in 0 or 1 < i, then it is an original observation
#     distance is in 2
#     number of entries in 3


scipy.cluster.hierarchy.dendrogram(cluster)

print 'saving...'
clusterdists = cluster[:,2]
plt.savefig('x_dendro.pdf', bbox_inches='tight')
plt.close()
plt.hist(clusterdists)
plt.savefig('x_clusterdists.pdf', bbox_inches='tight')
plt.close()

#from samplers.svm import svmnest
# svmnest.find_members
# svmnest.cut_cluster

# try a criterion
i = 1
n = 5
def plot_cluster(indices, title):
	global i, n
	plt.subplot(n, 1, i)
	plt.ylabel(title)
	plt.plot(highpoints, indices, 'x')
	i += 1
	#indices.max() + 3

plt.figure(figsize=(5, 20))
plot_cluster(scipy.cluster.hierarchy.fcluster(cluster, t=2, criterion='maxclust'), '2 clusters')
plot_cluster(scipy.cluster.hierarchy.fcluster(cluster, t=3, criterion='maxclust'), '3 clusters')
plot_cluster(scipy.cluster.hierarchy.fcluster(cluster, t=4, criterion='maxclust'), '4 clusters')
#plot_cluster(scipy.cluster.hierarchy.fcluster(cluster, 
#	t=(clusterdists.max() + numpy.median(clusterdists))/2,
#	criterion='distance'), 'middle break clustering')
plot_cluster(cut_cluster(tree, distances, scipy.stats.mstats.mquantiles(clusterdists, 0.1)*20 + clusterdists.max()/2), 'middle break clustering')
print scipy.stats.mstats.mquantiles(clusterdists, 0.99)
#plot_cluster(scipy.cluster.hierarchy.fcluster(cluster, 
#	t=scipy.stats.mstats.mquantiles(clusterdists, 0.9),
#	criterion='distance'), '90% quantile clustering')
plot_cluster(cut_cluster(tree, distances, scipy.stats.mstats.mquantiles(clusterdists, 0.99)), 'quantile break clustering')

plt.savefig('x_clustersplit.pdf', bbox_inches='tight')
plt.close()


