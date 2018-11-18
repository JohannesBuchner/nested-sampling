from . import clusterdetect 
import matplotlib.pyplot as plt
import numpy
from numpy import log10, log, exp
import scipy.spatial, scipy.cluster, scipy.stats
import itertools

def test_clusterdetect():
	u = numpy.random.uniform(0, 1, size=100).reshape((-1, 1))
	# detect clusters using hierarchical clustering
	distances = scipy.spatial.distance.cdist(u, u)
	cluster = scipy.cluster.hierarchy.single(distances)
	clusterdists = cluster[:,2]
	threshold = scipy.stats.mstats.mquantiles(clusterdists, 0.1)*10 + clusterdists.max()/3
	assigned = clusterdetect.cut_cluster(cluster, distances, threshold)
	clusterids = sorted(set(assigned))

	assert len(cluster) == len(distances) - 1
	n = len(cluster) + 1
	tree = {}
	fout = open('x_1d_clustergraph.dot', 'w')
	fout.write("digraph g{\n")
	for i, (a, b, dist, entries) in enumerate(cluster):
		a, b = int(a), int(b)
		#print('%3d%s %3d%s %.2f %3d' % (a, 
		#	'*' if a < n else ' ', b, '*' if b < n else ' ', dist, entries))
		if a < n:
			fout.write("%d[label=%.3f,shape=square];\n" % (a, u[a,0]))
		if b < n:
			fout.write("%d[label=%.3f,shape=square];\n" % (b, u[b,0]))
		fout.write("%d -> %d [label=%.2f];\n" % (i+n, a, dist))
		fout.write("%d -> %d [label=%.2f];\n" % (i+n, b, dist))
		tree[i + n] = (a, b, dist, entries)
	fout.write("}\n")
	# from cluster index 0 and 1 are combined in row i to form index n + i
	#     if index in 0 or 1 < i, then it is an original observation
	#     distance is in 2
	#     number of entries in 3


	scipy.cluster.hierarchy.dendrogram(cluster)

	print('saving...')
	clusterdists = cluster[:,2]
	plt.savefig('x_1d_dendro.pdf', bbox_inches='tight')
	plt.close()
	plt.hist(clusterdists)
	plt.savefig('x_1d_clusterdists.pdf', bbox_inches='tight')
	plt.close()

