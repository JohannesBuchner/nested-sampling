import numpy
import scipy.spatial, scipy.cluster
import matplotlib.pyplot as plt
from nested_sampling.clustering.neighbors import find_rdistance, is_within_distance_of, count_within_distance_of, any_within_distance_of
from nested_sampling.clustering.jarvispatrick import jarvis_patrick_clustering, jarvis_patrick_clustering_iterative
from nested_sampling.clustering.sdml import IdentityMetric, SimpleScaling, TruncatedScaling, MahalanobisMetric, TruncatedMahalanobisMetric, SDML
from collections import defaultdict

class ClusterResult(object):
	def __init__(self, points, clusters, metric, verbose=False):
		self.ws = points
		self.clusters = clusters
		self.metric = metric
		if verbose:
			print 'CLUSTERS:'
			for cluster in clusters:
				clusterpoints = metric.untransform(points[cluster,:])
				print 'CLUSTER:', clusterpoints.mean(axis=0), clusterpoints.std(axis=0)
	
	def get_cluster_id(self, point):
		w = self.metric.transform(point)
		dists = scipy.spatial.distance.cdist(self.ws, [w], metric='euclidean')
		i = numpy.argmin(dists)
		for j, cluster in enumerate(self.clusters):
			if i in cluster:
				return j
	
	def get_cluster_ids(self, points):
		ws = self.metric.transform(points)
		dists = scipy.spatial.distance.cdist(self.ws, ws, metric='euclidean')
		i = numpy.argmin(dists, axis=0)
		assert len(i) == len(points)
		results = []
		for ii in i:
			for j, cluster in enumerate(self.clusters):
				if ii in cluster:
					results.append(j)
		return results
	
	def get_n_clusters(self):
		return len(self.clusters)

class RadFriendsRegion(object):
	def __init__(self, members, maxdistance=None, metric='euclidean', nbootstraps=50, verbose=False):
		self.members = members
		assert metric == 'euclidean'
		if maxdistance is None:
			maxdistance = find_rdistance(members, nbootstraps=nbootstraps, 
				metric=metric, verbose=verbose)
			# print 'new RadFriendsRegion with r=', maxdistance
		self.maxdistance = maxdistance
		self.metric = metric
		self.verbose = verbose
		self.lo = numpy.min(self.members, axis=0) - self.maxdistance
		self.hi = numpy.max(self.members, axis=0) + self.maxdistance
	
	def add_members(self, us):
		self.members = numpy.vstack((self.members, us))
		self.lo = numpy.min(self.members, axis=0) - self.maxdistance
		self.hi = numpy.max(self.members, axis=0) + self.maxdistance
	
	def are_near_members(self, us):
		dists = scipy.spatial.distance.cdist(self.members, us, metric=self.metric)
		dist_criterion = dists < self.maxdistance
		return dist_criterion
	
	def count_nearby_members(self, us):
		return count_within_distance_of(self.members, self.maxdistance, us)
	
	def get_nearby_member_ids(self, u):
		return numpy.where(self.are_near_members([u]))[0]
	
	def is_inside(self, u):
		# is it true for at least one?
		if not ((u >= self.lo).all() and (u <= self.hi).all()):
			return False
		return is_within_distance_of(self.members, self.maxdistance, u)
		#return self.are_near_members([u]).any()
	
	def are_inside(self, us):
		# is it true for at least one?
		#return self.are_near_members(us).any(axis=0)
		return any_within_distance_of(self.members, self.maxdistance, us)
	
	def get_clusters(self):
		# agglomerate clustering of members
		dists = scipy.spatial.distance.cdist(self.members, self.members, metric=self.metric)
		connected = dists < self.maxdistance
		nmembers = len(self.members)
		cluster = {i:i for i in range(nmembers)}
		for i in range(nmembers):
			neighbors = numpy.where(connected[i,:])[0] #[i+1:]
			for j in neighbors:
				cluster[j] = cluster[i]
		result = defaultdict(list)
		for element, cluster_nro in cluster.items():
			result[cluster_nro].append(element)
		#print 'RadFriends: %d clusters' % len(result)
		return result
		
	
	def generate(self, nmax=0):
		members = self.members
		maxdistance = self.maxdistance
		nmembers, ndim = numpy.shape(self.members)
		# how many points to try to generate
		# if too small, many function calls, inefficient
		# if too large, large cdist matrices, spikes in memory use
		N = 1000
		verbose = self.verbose
		ntotal = 0
		#print 'draw from radfriends'
		while nmax == 0 or ntotal < nmax:
			# draw from box
			# this can be efficient if there are a lot of points
			ntotal = ntotal + N
			us = numpy.random.uniform(self.lo, self.hi, size=(N, ndim))
			mask = self.are_inside(us)
			#print 'accepted %d/%d [box draw]' % (mask.sum(), N)
			if mask.any():
				yield us[mask,:], ntotal
				#for u in us[mask,:]:
				#	#print 'box draw success:', ntotal
				#	yield u, ntotal
				ntotal = 0
			
			# draw from points
			# this can be efficient in higher dimensions
			us = members[numpy.random.randint(0, len(members), N),:]
			ntotal = ntotal + N
			if verbose: print 'chosen point', us
			# draw direction around it
			direction = numpy.random.normal(0, 1, size=(N, ndim))
			direction = direction / ((direction**2).sum(axis=1)**0.5).reshape((-1,1))
			if verbose: print 'chosen direction', direction
			# choose radius: volume gets larger towards the outside
			# so give the correct weight with dimensionality
			radius = maxdistance * numpy.random.uniform(0, 1, size=(N,1))**(1./ndim)
			us = us + direction * radius
			#mask = numpy.logical_and((u >= self.lo).all(axis=0), (u <= self.hi).all(axis=0))
			#if not mask.any():
			#	if verbose: print 'rejection because outside'
			#	continue
			#us = us[mask,:]
			#if verbose: print 'using point', us
			# count the number of points this is close to
			nnear = self.count_nearby_members(us)
			if verbose: print 'near', nnear
			# accept with probability 1./nnear
			coin = numpy.random.uniform(size=len(us))
			
			accept = coin < 1. / nnear
			#print 'accepted %d/%d [point draw]' % (accept.sum(), N)
			if not accept.any():
				if verbose: print 'probabilistic rejection due to overlaps'
				continue
			#print '  overlaps accepted %d of %d, typically %.2f neighbours' % (accept.sum(), N, nnear.mean())
			us = us[accept,:]
			yield us, ntotal
			#for u in us:
			#	#print 'ball draw success:', ntotal
			#	yield u, ntotal
			ntotal = 0


class MetricLearningFriendsConstrainer(object):
	"""
	0) Store unit metric.
	1) Splits live points into clusters using Jarvis-Patrick K=1 clustering
	2) Project new clusters onto old clusters for identification tree.
	   If new cluster encompasses more than one old cluster: 
	3) Overlay all clusters (shift by cluster mean) and compute new metric (covariance)
	4) Using original points and new metric, compute RadFriends bootstrapped distance and store
	5) In each RadFriends cluster, find points.
        6) If still mono-mode: no problem
	   If discovered new clusters in (1): store filtering function and cluster assignment
	   If no new clusters: no problem
	
	When point is replaced:
	1) Check if point is in a cluster that is dying out: 
	   when point is last in current or previously stored clustering
	
	For sampling:
	1) Draw a new point from a metric-shaped ball from random point
	2) Filter with previous filtering functions if exist
	3) Evaluate likelihood
	
	For filtering:
	1) Given a point, check if within metric-shaped ball of a existing point
	2) Filter with previous filtering functions if exist
	
	"""
	def __init__(self, metriclearner, rebuild_every = 50, verbose = False,
			keep_phantom_points=False, optimize_phantom_points=False,
			force_shrink=False):
		self.iter = 0
		self.region = None
		self.rebuild_every = int(rebuild_every)
		self.previous_filters = []
		self.verbose = verbose
		self.keep_phantom_points = keep_phantom_points
		self.optimize_phantom_points = optimize_phantom_points
		self.force_shrink = force_shrink
		self.phantom_points = []
		self.phantom_points_Ls = []
		self.metriclearner = metriclearner
		self.metric = IdentityMetric()
		self.clusters = None
		self.direct_draws_efficient = True
	
	def cluster(self, u, ndim, keepMetric=False):
		"""
		1) Splits live points into clusters using Jarvis-Patrick K=1 clustering
		2) Project new clusters onto old clusters for identification tree.
		   If new cluster encompasses more than one old cluster: 
		3) Overlay all clusters (shift by cluster mean) and compute new metric (covariance)
		4) Using original points and new metric, compute RadFriends bootstrapped distance and store
		5) In each RadFriends cluster, find points.
		6) If still mono-mode: no problem
		   If discovered new clusters in (1): store filtering function and cluster assignment
		   If no new clusters: no problem
		"""
		w = self.metric.transform(u)
		prev_region = self.region
		if keepMetric:
			self.region = RadFriendsRegion(members=w)
			if self.force_shrink and self.region.maxdistance > prev_region.maxdistance:
				self.region = RadFriendsRegion(members=w, maxdistance=prev_region.maxdistance)
			return
		
		metric_updated = False
		clustermetric = self.metric
		if self.verbose: print 'computing distances for clustering...'
		wdists = scipy.spatial.distance.cdist(w, w, metric='euclidean')
		# apply Jarvis-Patrick clustering
		if self.verbose: print 'Clustering...'
		clusters = jarvis_patrick_clustering_iterative(wdists, number_of_neighbors=len(wdists), n_stable_iterations=3)
		# Overlay all clusters (shift by cluster mean) 
		if self.verbose: print 'Metric update ...'
		shifted_cluster_members = []
		for members in clusters:
			cluster_mean = numpy.mean(u[members,:], axis=0)
			shifted_cluster_members += (u[members,:] - cluster_mean).tolist()
		shifted_cluster_members = numpy.asarray(shifted_cluster_members)
		# Using original points and new metric, compute RadFriends bootstrapped distance and store
		if self.metriclearner == 'none':
			metric = self.metric # stay with identity matrix
			metric_updated = False
		elif self.metriclearner == 'simplescaling' or (self.metriclearner == 'mahalanobis' and ndim == 1):
			metric = SimpleScaling()
			metric.fit(shifted_cluster_members)
			metric_updated = True
		elif self.metriclearner == 'truncatedscaling' or (self.metriclearner == 'truncatedmahalanobis' and ndim == 1):
			metric = TruncatedScaling()
			metric.fit(shifted_cluster_members)
			metric_updated = self.metric == IdentityMetric() or not numpy.all(self.metric.scale == metric.scale)
		elif self.metriclearner == 'mahalanobis':
			metric = MahalanobisMetric()
			metric.fit(shifted_cluster_members)
			metric_updated = True
		elif self.metriclearner == 'truncatedmahalanobis':
			metric = TruncatedMahalanobisMetric()
			metric.fit(shifted_cluster_members)
			metric_updated = self.metric == IdentityMetric() or not (numpy.all(self.metric.scale == metric.scale) and numpy.all(self.metric.cov == metric.cov))
		elif self.metriclearner == 'sdml':
			metric = SDML()
			metric.fit(shifted_cluster_members, W = numpy.ones((len(w), len(w))))
			metric_updated = True
		else:
			assert False, self.metriclearner
		
		self.metric = metric
		
		oldclusters = self.clusters
		self.clusters = clusters
		
		wnew = self.metric.transform(u)
		#shifted_cluster_members = []
		#for members in clusters:
		#	cluster_mean = numpy.mean(wnew[members,:], axis=0)
		#	shifted_cluster_members += (wnew[members,:] - cluster_mean).tolist()
		#shifted_cluster_members = numpy.asarray(shifted_cluster_members)
		#shifted_region = RadFriendsRegion(members=shifted_cluster_members)
		if self.verbose: print 'Region update ...'
		
		self.region = RadFriendsRegion(members=wnew) #, maxdistance=shifted_region.maxdistance)
		if not metric_updated and self.force_shrink and prev_region is not None:
			if self.region.maxdistance > prev_region.maxdistance:
				self.region = RadFriendsRegion(members=w, maxdistance=prev_region.maxdistance)
		
		if oldclusters is None or len(clusters) != len(oldclusters):
		#if True:
			# store filter function
			self.previous_filters.append((self.metric, self.region, ClusterResult(metric=clustermetric, clusters=self.clusters, points=w)))
		
		#rfclusters = self.region.get_clusters()
		#print 'Clustering: JP has %d clusters, radfriends has %d cluster:' % (len(clusters), len(rfclusters))
		#var = self.iter, self.metric, u, self.region.maxdistance
		#assert self.is_inside(numpy.array([0.123456]*ndim)), var
		#assert self.is_inside(numpy.array([0.654321]*ndim)), var
		if self.verbose: print 'Metric+region update done.'
	
	def are_inside_cluster(self, points):
		w = self.metric.transform(points)
		return self.region.are_inside(w)
	
	def is_inside(self, point):
		if not ((point >= 0).all() and (point <= 1).all()):
			return False
		w = self.metric.transform(point)
		return self.region.is_inside(w)

	def generate(self, ndim):
		ntotal = 0
		"""
		for w, n in self.region.generate():
			u = self.metric.untransform(w)
			ntotal += n
			#if numpy.all(u >= 0) and numpy.all(u <= 1):
			if all([0 <= ui <= 1 for ui in u]):
				yield u, ntotal
				ntotal = 0
			else:
				print 'rejected [box constraint]'
			
		"""
		N = 10000
		while True:
			#if numpy.random.uniform() < 0.01:
			if True:
				# draw from unit cube
				# this can be efficient if volume still large
				ntotal = ntotal + N
				us = numpy.random.uniform(size=(N, ndim))
				ws = self.metric.transform(us)
				nnear = self.region.are_inside(ws)
				#print '  %d of %d accepted' % (nnear.sum(), N)
				for u in us[nnear,:]:
					#print 'unit cube draw success:', ntotal
					yield u, ntotal
					ntotal = 0
			if ndim < 40:
				# draw from radfriends directly
				for ws, n in self.region.generate(N):
					us = self.metric.untransform(ws)
					assert us.shape[1] == ndim, us.shape
					ntotal = ntotal + n
					mask = numpy.logical_and(us < 1, us > 0).all(axis=1)
					assert mask.shape == (len(us),), (mask.shape, us.shape)
					if mask.any():
						#print 'radfriends draw in unit cube:', mask.sum(), ntotal
						for u in us[mask,:]:
							assert u.shape == (us[0].shape), (u.shape, us.shape, mask.shape)
							yield u, ntotal
							ntotal = 0
					#if all([0 <= ui <= 1 for ui in u]):
					#	yield u, ntotal
					#	ntotal = 0
		
	def rebuild(self, u, ndim, keepMetric=False):
		self.cluster(u=u, ndim=ndim, keepMetric=keepMetric)
		if len(self.phantom_points) > 0:
			print 'adding phantom points to radfriends:', self.phantom_points
			self.region.add_members(self.metric.transform(self.phantom_points))
			#self.region.members = numpy.vstack((self.region.members, 
			#	self.metric.transform(self.phantom_points)))
		# reset generator
		if self.verbose: print 'maxdistance:', self.region.maxdistance
		self.generator = self.generate(ndim)
	
	def is_last_of_its_cluster(self, u, uothers):
		# check if only point of current clustering left
		w = self.metric.transform(u)
		wothers = self.metric.transform(uothers)
		othersregion = RadFriendsRegion(members=wothers, maxdistance=self.region.maxdistance)
		if not othersregion.is_inside(w):
			return True
		
		# check previous clusterings
		for metric, region, clusters in self.previous_filters:
			if clusters.get_n_clusters() < 2:
				# only one cluster, so can not die out
				continue
			
			# check in which cluster this point was
			i = clusters.get_cluster_id(u)
			j = clusters.get_cluster_ids(uothers)
			#print 'cluster_sets:', i, set(j)
			if i not in j:
				# this is the last point of that cluster
				return True
		return False
	
	def _draw_constrained_prepare(self, Lmin, priortransform, loglikelihood, live_pointsu, ndim, **kwargs):
		self.iter += 1
		rebuild = self.iter % self.rebuild_every == 1 or self.region is None
		if rebuild:
			self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=False)
		assert self.generator is not None
		ntoaccept = 0
		ntotalsum = 0
		if self.keep_phantom_points:
			# check if the currently dying point is the last of a cluster
			starti = kwargs['starti']
			ucurrent = live_pointsu[starti]
			#wcurrent = self.metric.transform(ucurrent)
			uothers = [ui for i, ui in enumerate(live_pointsu) if i != starti]
			#wothers = self.metric.transform(uothers)
			phantom_points_added = False
			if self.is_last_of_its_cluster(ucurrent, uothers):
				if self.optimize_phantom_points:
					print 'optimizing phantom point', ucurrent
					import scipy.optimize
					def f(u):
						w = self.metric.transform(u)
						if not self.region.is_inside(w):
							return 1e100
						x = priortransform(u)
						L = loglikelihood(x)
						if self.verbose: print 'OPT %.2f ' % L, u
						return -L
					r = scipy.optimize.fmin(f, ucurrent, ftol=0.5, full_output=True)
					ubest = r[0]
					Lbest = -r[1]
					ntoaccept += r[3]
					print 'optimization gave', r
					wbest = self.metric.transform(ubest)
					if not self.is_last_of_its_cluster(ubest, uothers):
						print 'that optimum is inside the other points, so no need to store'
					else:
						print 'remembering phantom point', ubest, Lbest
						self.phantom_points.append(ubest)
						self.phantom_points_Ls.append(Lbest)
						phantom_points_added = True
				else:
					print 'remembering phantom point', ucurrent
					self.phantom_points.append(ucurrent)
					phantom_points_added = True
			
			if phantom_points_added:
				self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=False)
				rebuild = True
			
			
			if self.optimize_phantom_points and len(self.phantom_points) > 0:
				# purge phantom points that are below Lmin
				keep = [i for i, Lp in enumerate(self.phantom_points_Ls) if Lp > Lmin]
				self.phantom_points = [self.phantom_points[i] for i in keep]
				if len(keep) != len(self.phantom_points_Ls):
					print 'purging some old phantom points. new:', self.phantom_points, Lmin
					self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=False)
					rebuild = True
					
				self.phantom_points_Ls = [self.phantom_points_Ls[i] for i in keep]
		return ntoaccept, ntotalsum, rebuild
	
	def get_Lmax(self):
		if len(self.phantom_points_Ls) == 0:
			return None
		return max(self.phantom_points_Ls)

	def draw_constrained(self, Lmin, priortransform, loglikelihood, live_pointsu, ndim, **kwargs):
		ntoaccept, ntotalsum, rebuild = self._draw_constrained_prepare(Lmin, priortransform, loglikelihood, live_pointsu, ndim, **kwargs)
		rebuild_metric = rebuild
		while True:
			for u, ntotal in self.generator:
				assert (u >= 0).all() and (u <= 1).all(), u
				ntotalsum += ntotal
				x = priortransform(u)
				L = loglikelihood(x)
				ntoaccept += 1

				#print 'ntotal:', ntotal
				if ntotal > 100000:
					self.direct_draws_efficient = False
				
				if L > Lmin:
					# yay, we win
					return u, x, L, ntoaccept
				
				# if running very inefficient, optimize clustering 
				#     if we haven't done so at the start
				if not rebuild and ntoaccept > 1000:
					rebuild = True
					print 'low efficiency is triggering RadFriends rebuild'
					self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=True)
					break
				#if not rebuild_metric and ntoaccept > 1000:
				#	rebuild_metric = True
				#	print 'low efficiency is triggering metric rebuild'
				#	self.rebuild(numpy.asarray(live_pointsu), ndim, keepMetric=False)
				#	break
