import scipy, scipy.stats, scipy.cluster
import numpy
from numpy import exp, log, log10

def find_members(tree, c, members = []):
	a, b, dist, entries = tree[c]
	if a in tree:
		find_members(tree, a, members)
	else:
		members.append(a)
	if b in tree:
		find_members(tree, b, members)
	else:
		members.append(b)


# start from bottom while distance below threshold
def cut_cluster(cluster, distances, threshold, minsize=30, verbose=1):
	n = len(distances)
	assert len(cluster) == n - 1
	tree = {}
	for i, (a, b, distance, entries) in enumerate(cluster):
		tree[i + n] = (int(a), int(b), distance, entries)
	
	isparent = numpy.zeros(n*2 - 1) == 0
	for i, (a, b, dist, entries) in tree.iteritems():
		nchildren = min(cluster[a - n][3] if a >= n else 1, cluster[b - n][3] if a >= n else 1)
		if nchildren <= minsize or (isparent[a] and isparent[b] and dist < threshold):
			# node is too small so far, or children are near
			# children are not parent, we are parent
			isparent[a] = False
			isparent[b] = False
		else:  # we are too far away to claim parent
			isparent[i] = False
			print '    splitting with %d, %d children' % (cluster[a - n][3] if a >= n else 1, cluster[b - n][3] if a >= n else 1)
	# go through parents
	if verbose > 0 and isparent.sum() > 1:
		print 'cut at %.2f found %d clusters' % (threshold, isparent.sum())
	ids = set(numpy.arange(n*2 - 1)[isparent])
	if verbose > 1:
		print '  parents:', ids
		for i, (a, b, dist, entries) in tree.iteritems():
			if i in ids or a in ids or b in ids:
				print '    %3d: %3d%s %3d%s %.2f %3d' % (i, a, 
					'*' if a < n else ' ', b, '*' if b < n else ' ', dist, entries)
	assigned = numpy.zeros(n) - 1
	for i, c in enumerate(ids):
		if c not in tree:
			members = [c]
			#print i, c, 'singular', len(members)
		else:
			members = []
			find_members(tree, c, members)
			#print i, c, tree[c][3], len(members)
		for m in members:
			assigned[m] = i
	return assigned	

