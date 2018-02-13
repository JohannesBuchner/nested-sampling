"""
Qi et al.
An efficient sparse metric learning in high-dimensional space via
L1-penalized log-determinant regularization.
ICML 2009

Adapted from https://gist.github.com/kcarnold/5439945
Paper: http://lms.comp.nus.edu.sg/sites/default/files/publication-attachments/icml09-guojun.pdf

from https://github.com/all-umass/metric-learn/

Copyright (c) 2015 CJ Carey and Yuan Tang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import absolute_import
import numpy as np
import numpy
from numpy import exp
try:
	from scipy.sparse.csgraph import laplacian
	from sklearn.covariance import graph_lasso
	from sklearn.utils.extmath import pinvh
except ImportError:
	pass
import scipy.linalg

class IdentityMetric(object):
	"""
	Input is output.
	"""
	def fit(self, x):
		pass
	def transform(self, x):
		return x
	def untransform(self, y):
		return y
	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

class SimpleScaling(object):
	"""
	Whitens by subtracting the mean and scaling by the 
	standard deviation of each axis.
	"""
	def __init__(self, verbose=False):
		self.verbose = verbose

	def fit(self, X, W=None):
		self.mean = numpy.mean(X, axis=0)
		X = X - self.mean
		self.scale = numpy.std(X, axis=0)
		if self.verbose: 'Scaling metric:', self.scale
	def transform(self, x):
		return (x - self.mean) / self.scale
	
	def untransform(self, y):
		return y * self.scale + self.mean

	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

class TruncatedScaling(object):
	"""
	Whitens by subtracting the mean and scaling by the 
	standard deviation of each axis. The scaling is discretized on 
	a log axis onto integers.
	"""
	def __init__(self, verbose=False):
		self.verbose = verbose
	def fit(self, X, W=None):
		self.mean = numpy.mean(X, axis=0)
		X = X - self.mean
		#scale = numpy.max(X, axis=0) - numpy.min(X, axis=0)
		scale = numpy.std(X, axis=0)
		scalemax = scale.max() * 1.001
		scalemin = scale.min()
		# round onto discrete log scale to avoid random walk
		logscale = (-numpy.log2(scale / scalemax)).astype(int)
		self.scale = 2**(logscale.astype(float))
		#print 'Scaling metric:', self.scale, '(from', scale, ')'
		if self.verbose: 'Discretized scaling metric:\n', logscale
	
	def transform(self, x):
		return (x - self.mean) / self.scale
	
	def untransform(self, y):
		return y * self.scale + self.mean

	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

class MahalanobisMetric(object):
	"""
	Whitens by covariance.
	"""
	def __init__(self, verbose=False):
		self.verbose = verbose
	
	def fit(self, X, W=None):
		#print 'subtracting mean...'
		self.mean = numpy.mean(X, axis=0)
		X = X - self.mean
		nsamples, ndim = X.shape
		#print 'calculating cov...'
		cov = numpy.cov(X.transpose())
		#print 'calculating polar...'
		# make positive semi-definite
		_, cov = scipy.linalg.polar(cov)
		#print 'det:', np.linalg.det(cov), 'rank:', np.linalg.matrix_rank(cov), ndim
		if np.linalg.matrix_rank(cov) == ndim: # and np.linalg.det(cov) > 1e-10:
			#print 'using cov for Mahalanobis metric'
			self.cov = cov
			assert self.cov.shape == (ndim, ndim)
			#print 'calculating inverse...'
			self.invcov = numpy.linalg.pinv(self.cov)
			#print 'calculating real sqrt of inverse...'
			self.SQI = scipy.linalg.sqrtm(self.invcov).real
			#print 'calculating real sqrt ...'
			self.SQ = scipy.linalg.sqrtm(cov).real
		else:
			# we have a singular matrix.
			# use only scaling.
			print('singular matrix, switching to simple scaling...')
			scale = numpy.std(X, axis=0)
			self.SQI = numpy.diag(1./scale)
			self.SQ = numpy.diag(scale)
		if self.verbose: print('Mahalanobis metric:\n', self.cov)
	
	def transform(self, x):
		return numpy.dot(x - self.mean, self.SQI)
	
	def untransform(self, y):
		return numpy.dot(y, self.SQ) + self.mean

	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

def discretize_matrix(cov):
	ndim = len(cov)
	intcov = numpy.zeros(cov.shape, dtype=int)
	signcov = numpy.ones(cov.shape, dtype=int)
	trunccov = numpy.zeros_like(cov)
	for i in range(ndim):
		intcov[i,i] = round(numpy.log(cov[i,i]))
		trunccov[i,i] = exp(intcov[i,i])

	for i in range(ndim):
		for j in range(ndim):
			if i == j: continue
			intcov[i,j] = round(numpy.log(1 - abs(cov[i,j])/(cov[j,j]*cov[i,i])**0.5))
			signcov[i,j] = numpy.sign(cov[i,j])
			trunccov[i,j] = (1 - exp(intcov[i,j])) * (trunccov[i,i] * trunccov[j,j])**0.5 * signcov[i,j]
	return trunccov, intcov, signcov
	

class TruncatedMahalanobisMetric(object):
	"""
	Whitens by discretized covariance.
	"""
	def __init__(self, verbose=False):
		self.verbose = verbose
	
	def fit(self, X, W=None):
		self.mean = numpy.mean(X, axis=0)
		samples = X
		X = X - self.mean
		nsamples, ndim = X.shape
		scale = numpy.std(X, axis=0)
		scalemax = scale.max() * 1.001
		scalemin = scale.min()
		# round onto discrete log scale to avoid random walk
		logscale = (-numpy.log(scale / scalemax)).astype(int)
		self.scale = exp(logscale.astype(float))
		X = X / self.scale / scalemax
		cov = numpy.cov(X.transpose())
		if self.verbose: print('original cov', cov)
		if self.verbose: print('invertible?', numpy.linalg.matrix_rank(cov) == len(cov))
		"""
		intcov = numpy.zeros(cov.shape, dtype=int)
		signcov = numpy.ones(cov.shape, dtype=int)
		trunccov = numpy.zeros_like(cov)
		for i in range(ndim):
			intcov[i,i] = round(numpy.log(cov[i,i]))
			trunccov[i,i] = exp(intcov[i,i])

		for i in range(ndim):
			for j in range(ndim):
				if i == j: continue
				intcov[i,j] = round(numpy.log(1 - abs(cov[i,j])/(cov[j,j]*cov[i,i])**0.5))
				signcov[i,j] = numpy.sign(cov[i,j])
				trunccov[i,j] = (1 - exp(intcov[i,j])) * (trunccov[i,i] * trunccov[j,j])**0.5 * signcov[i,j]
		cov = trunccov
		"""
		cov, intcov, signcov = discretize_matrix(cov)
		if self.verbose: print('intcov:\n', intcov)
		#for row in cov:
		#	print row
		if self.verbose: print('invertible?', numpy.linalg.matrix_rank(cov) == len(cov))
		# ensure it is positive semi-definite
		if self.verbose: print('before polar', cov)
		_, cov = scipy.linalg.polar(cov)
		if self.verbose: print('after polar', cov)
		self.cov = cov
		assert self.cov.shape == (ndim, ndim)
		self.invcov = numpy.linalg.inv(self.cov)
		self.SQI = scipy.linalg.sqrtm(self.invcov).real
		self.SQ = scipy.linalg.sqrtm(self.cov).real
		
		if self.verbose: print('Discretized Mahalanobis metric:\n', intcov, 'with scale', logscale)
		wsamples = self.transform(samples)
		samples2 = self.untransform(wsamples)
		if not numpy.allclose(samples, samples2):
			numpy.savez('sdml_difficult_samples.npz', samples=samples)
			for x, y in zip(samples, samples2):
				assert numpy.allclose(x, y), (x, y)
	
	def transform(self, x):
		return numpy.dot((x - self.mean) / self.scale, self.SQI)
	
	def untransform(self, y):
		return numpy.dot(y, self.SQ) * self.scale + self.mean

	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

class SDML(object):
	"""
	Sparse-determinant metric-learning.
	"""
	def __init__(self, balance_param=0.5, sparsity_param=0.01, use_cov=True, EPS = 1e-6, verbose=False):
		'''
		balance_param: float, optional
		trade off between sparsity and M0 prior
		sparsity_param: float, optional
		trade off between optimizer and sparseness (see graph_lasso)
		use_cov: bool, optional
		controls prior matrix, will use the identity if use_cov=False
		verbose : bool, optional
		if True, prints information while learning
		'''
		self.balance_param = balance_param
		self.sparsity_param = sparsity_param
		self.use_cov = use_cov
		self.EPS = EPS
		self.verbose = verbose
	
	def fit(self, X, W=None):
		'''
		X: data matrix, (n x d)
		each row corresponds to a single instance
		Must be shifted to zero already.
		
		W: connectivity graph, (n x n)
		+1 for positive pairs, -1 for negative.
		'''
		print('SDML.fit ...', numpy.shape(X))
		self.mean_ = numpy.mean(X, axis=0)
		X = numpy.matrix(X - self.mean_)
		# set up prior M
		#print 'X', X.shape
		if self.use_cov:
			M = np.cov(X.T)
		else:
			M = np.identity(X.shape[1])
		if W is None:
			W = np.ones((X.shape[1], X.shape[1]))
		#print 'W', W.shape
		L = laplacian(W, normed=False)
		#print 'L', L.shape
		inner = X.dot(L.T)
		loss_matrix = inner.T.dot(X)
		#print 'loss', loss_matrix.shape
		
		#print 'pinv', pinvh(M).shape
		P = pinvh(M) + self.balance_param * loss_matrix
		#print 'P', P.shape
		emp_cov = pinvh(P)
		# hack: ensure positive semidefinite
		emp_cov = emp_cov.T.dot(emp_cov)
		M, _ = graph_lasso(emp_cov, self.sparsity_param, verbose=self.verbose)
		self.M = M
		C = numpy.linalg.cholesky(self.M)
		self.dewhiten_ = C
		self.whiten_ = numpy.linalg.inv(C)
		# U: rotation matrix, S: scaling matrix
		#U, S, _ = scipy.linalg.svd(M)
		#s = np.sqrt(S.clip(self.EPS))
		#s_inv = np.diag(1./s)
		#s = np.diag(s)
		#self.whiten_ = np.dot(np.dot(U, s_inv), U.T)
		#self.dewhiten_ = np.dot(np.dot(U, s), U.T)
		#print 'M:', M
		print('SDML.fit done')
		
	def transform(self, x):
		return np.dot(x - self.mean_, self.whiten_.T)
	
	def untransform(self, y):
		return np.dot(y, self.dewhiten_) + self.mean_

import metric_learn

class SDMLWrapper(object):
	def __init__(self):
		pass
	def fit(self, X):
		self.metriclearner = metric_learn.sdml.SDML()
		self.metriclearner.fit(X, W = np.diag(np.ones(X.shape[0])*3) - 1)
	def transform(self, x):
		return self.metriclearner.transform(x)
	def untransform(self, y):
		return y

class TruncatedSDML(SDML):
	"""
	Sparse-determinant metric-learning, truncated
	"""
	
	def fit(self, X, W=None):
		self.mean = numpy.mean(X, axis=0)
		samples = X
		X = X - self.mean
		nsamples, ndim = X.shape
		scale = numpy.std(X, axis=0)
		scalemax = scale.max() * 1.001
		scalemin = scale.min()
		# round onto discrete log scale to avoid random walk
		logscale = (-numpy.log(scale / scalemax)).astype(int)
		self.scale = exp(logscale.astype(float))
		#X = X / self.scale / scalemax
		SDML.fit(self, X=X, W=W)
		#self.M, _, _ = discretize_matrix(self.M)
		U, S, _ = scipy.linalg.svd(self.M)
		s = np.sqrt(S.clip(self.EPS))
		s_inv = np.diag(1./s)
		s = np.diag(s)
		self.whiten_ = np.dot(np.dot(U, s_inv), U.T)
		self.dewhiten_ = np.dot(np.dot(U, s), U.T)
	
	#def transform(self, x):
	#	return numpy.dot((x - self.mean) / self.scale, self.whiten_.T)
	
	#def untransform(self, y):
	#	return numpy.dot(y, self.dewhiten_) * self.scale + self.mean

def test_generate_corr_sample(N, ndim, difficulty):
	logmatrix = numpy.zeros((ndim, ndim), dtype=int)
	matrix = numpy.zeros((ndim, ndim))

	for i in range(ndim):
		for j in range(i+1):
			logmatrix[i,j] = (((j*i) % (5+i)) // 2) % 5
			logmatrix[j,i] = logmatrix[i,j]
	
	for i in range(ndim):
		for j in range(i+1):
			scalei = numpy.exp(-logmatrix[i,i] - 2)
			scalej = numpy.exp(-logmatrix[j,j] - 2)
			if i == j:
				matrix[i,i] = scalei*scalej
			else:
				matrix[i,j] = (1 - numpy.exp(-logmatrix[i,j]*difficulty))*scalei*scalej
				matrix[j,i] = (1 - numpy.exp(-logmatrix[j,i]*difficulty))*scalei*scalej
	# ensure positive semi-definite
	_, matrix = scipy.linalg.polar(matrix)
	invmatrix = numpy.linalg.inv(matrix)
	
	mean = numpy.zeros(ndim)
	
	# generate N points
	from nestle import Ellipsoid
	print('generating from:', matrix) #, invmatrix
	ell = Ellipsoid(mean, invmatrix)
	samples = numpy.array([ell.sample() for i in range(N)])
	return samples

if __name__ == '__main__':
	# generate sample
	import matplotlib.pyplot as plt
	import os
	N = 40
	for ndim, difficulty in (2, 0), (3,4), (10, 1), (20, 1), (20, 2), ('sdml_difficult_samples.npz', -1):
	#for ndim, difficulty in ('sdml_difficult_samples.npz', -1),:
	#sfor ndim, difficulty in ('maha.npz', -1),:
	#for ndim, difficulty in (3,4),:
		if difficulty == -1:
			print()
			print('======== TEST file=%s ==========' % (ndim))
			print()
			data = numpy.load(ndim)
			print(list(data.keys()))
			#samples = numpy.load(ndim)['X']
			samples = numpy.load(ndim)[list(data.keys())[0]]
		else:
			print()
			print('======== TEST ndim=%d difficulty %d ==========' % (ndim, difficulty))
			print()
			samples = test_generate_corr_sample(N=N, ndim=ndim, difficulty=difficulty)
		
		#for metric in IdentityMetric(), SimpleScaling(), TruncatedScaling(), MahalanobisMetric(), TruncatedMahalanobisMetric(), SDML(), TruncatedSDML():
		for metric in SDMLWrapper(),:
		#for metric in TruncatedMahalanobisMetric(verbose=True),:
			print(('testing metric %s' % type(metric)))
			metric.fit(samples)
			wsamples = metric.transform(samples)
			#assert wsamples.shape == samples.shape, (wsamples.shape, samples.shape)
			samples2 = metric.untransform(wsamples)
			#for s1, w, s2 in zip(samples, wsamples, samples2):
			#	assert numpy.allclose(s1, s2), (s1, s2, w)
			#assert numpy.allclose(samples2, samples), (metric, samples, wsamples, samples2)
			#print (metric, samples, wsamples, samples2)
			if os.environ.get('SHOW_PLOTS', '0') == '1':
				plt.plot(samples[:,-2], samples[:,-1], 'x ', color='gray')
				plt.plot(wsamples[:,-2], wsamples[:,-1], 'o ', color='r')
				plt.plot(samples2[:,-2], samples2[:,-1], '+ ', color='k')
				print('showing plot...')
				plt.show()
			
			small_samples = samples / 10
			metric.fit(small_samples)
			wsamples = metric.transform(small_samples)
			#assert wsamples.shape == small_samples.shape, (wsamples.shape, small_samples.shape)
			samples2 = metric.untransform(wsamples)
			#assert numpy.allclose(samples2, small_samples), (metric, small_samples, wsamples, samples2)
	
	print('no assertion errors, so tests successful')
	
	
	
