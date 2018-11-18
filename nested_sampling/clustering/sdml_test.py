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
import numpy
from numpy import exp, log, log10
import scipy.linalg
from .sdml import IdentityMetric, SimpleScaling, TruncatedScaling, MahalanobisMetric, TruncatedMahalanobisMetric, SDML, TruncatedSDML, SDMLWrapper

def _test_generate_corr_sample(N, ndim, difficulty):
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

def test_sdml():
	N = 40
	for ndim, difficulty in (2, 0), (3,4), (10, 1), (20, 1), (20, 2):
		print()
		print('======== TEST ndim=%d difficulty %d ==========' % (ndim, difficulty))
		print()
		samples = _test_generate_corr_sample(N=N, ndim=ndim, difficulty=difficulty)
		for metric in IdentityMetric(), SimpleScaling(), TruncatedScaling(), MahalanobisMetric(), TruncatedMahalanobisMetric(), SDML(), TruncatedSDML(),:# SDMLWrapper(),:
			print(('testing metric %s' % type(metric)))
			metric.fit(samples)
			wsamples = metric.transform(samples)
			assert wsamples.shape == samples.shape, (wsamples.shape, samples.shape)
			samples2 = metric.untransform(wsamples)
			for s1, w, s2 in zip(samples, wsamples, samples2):
				assert numpy.allclose(s1, s2), (s1, s2, w)
			assert numpy.allclose(samples2, samples), (metric, samples, wsamples, samples2)
			
			small_samples = samples / 10
			metric.fit(small_samples)
			wsamples = metric.transform(small_samples)
			assert wsamples.shape == small_samples.shape, (wsamples.shape, small_samples.shape)
			samples2 = metric.untransform(wsamples)
			assert numpy.allclose(samples2, small_samples), (metric, small_samples, wsamples, samples2)
	
	print('no assertion errors, so tests successful')
	
	

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
			samples = _test_generate_corr_sample(N=N, ndim=ndim, difficulty=difficulty)
		
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
	
	
	
