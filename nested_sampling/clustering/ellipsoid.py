from __future__ import print_function
import scipy, scipy.optimize
import numpy
from numpy import log, exp, pi, log10

class EllipsoidContainer(object):
	def __init__(self, x):
		self.x = x
		xlow  = x.min(axis=0)
		xhigh = x.max(axis=0)
		xmid  = (xhigh + xlow)/2.
		rmax = ((((x - xmid)**2).sum())**0.5/len(xmid)).max()
		xwidth= (xhigh - xlow) / 2 * 0 + rmax * 2**0.5 / 2.
		self.cov = numpy.matrix(numpy.diag((xwidth)**2))
		self.mid = xmid
		print(self.cov, self.mid)
	def plot(self, **kwargs):
		x0, y0 = self.mid[0], self.mid[1]
		sx, sy = self.cov[0,0], self.cov[1,1]
		rho = self.cov[0,1]
		#xhi = (sx**2/(1 - rho**2))**0.5
		#xhi = x0 + (-sy * (rho**2 - sx*sy))**0.5 / (rho**2 - sx*sy)
		xlo = x0 - sx**0.5
		xhi = x0 + sx**0.5
		x = numpy.linspace(xlo, xhi, 400)
		#y = (sy**2*(sx**2 - x**2) / (sx * (2 * rho * sy * x**2 + sx)))**0.5
		#ymid = rho * sx * sy * x
		#ysol = (sx**2 * sy**2 * (rho**2 * x**2 + sx**2 - x**2))**0.5
		#ymid = -rho * (x - x0)
		#ysol = ( rho**2 * (x - x0)**2 - (sx * sy * (x - x0)**2) + sy)**0.5
		ymid = rho * (x - x0) + sx*y0
		print('rho', rho**2, sx*sy)
		ysol = (-(rho**2 - sx*sy) * (sx - (x - x0)**2) )**0.5
		plt.plot(x, (ymid + ysol) / sx, '-', **kwargs)
		plt.plot(x, (ymid - ysol) / sx, '-', **kwargs)
	
	def contains_points(self):
		invcov = self.cov.I
		#print 'invcov', invcov
		for xi in self.x:
			u = numpy.matrix(xi - self.mid).T
			#print 'contains:', xi, u
			d = u.T * invcov * u
			bad = numpy.abs(d) > 1
			if bad.any():
				print('some points not contained:')
				print(xi, d)
				return False
		return True
	def unpack(self, params):
		ndim = len(self.mid)
		
		self.mid = params[:ndim]
		k = ndim
		for i in range(ndim):
			for j in range(i, ndim):
				self.cov[i, j] = params[k]
				k = k + 1
	def pack(self):
		ndim = len(self.mid)
		params = list(self.mid)
		for i in range(ndim):
			for j in range(i, ndim):
				params.append(self.cov[i, j])
		return params
	
	def minfunc(self, params):
		# update parameters
		self.unpack(params)
		if (numpy.diag(self.cov) < 0).any():
			return 1e300
		#nparams = ndim + ndim * (ndim - 1) / 2
		# all points must be contained
		#plt.figure('progress', figsize=(5,5))
		#plt.plot(self.x[:,0] - self.mid[0], self.x[:,1], 'x', color='b')
		#self.plot(color='r', label='contains: %s' % self.contains_points())
		#plt.legend(loc='upper left')
		#plt.savefig('ellipse.pdf')
		#plt.close()
		if not self.contains_points():
			return 1e300
		# otherwise, we return the volume, which we want to minimize
		d = numpy.linalg.det(self.cov)
		if numpy.isnan(d):
			return 1e300
		#print d
		#V = 4/3 * pi * d**0.5
		V = d**0.5
		print('*', params)
		print('*', self.cov, self.mid)
		print('** V', V)
		return V
	def optimize(self):
		start = self.pack()
		#print start
		#print 'V:', self.minfunc(start)
		##self.cov[0,0] *= 0.8
		#self.mid[1] = +1.1
		#self.cov[1,1] *= 0.7
		#self.plot(color='r')
		#start2 = self.pack()
		#print 'V:', self.minfunc(start2)
		result = scipy.optimize.fmin(self.minfunc, start)
		self.unpack(result)
		assert self.contains_points()
		print(result)



if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	mean = [30, -30]
	cov = [[1, 2], [2, 2]]
	numpy.random.seed(2)
	points = numpy.random.multivariate_normal(mean, cov, size=100)
	plt.figure(figsize=(5,5))
	ell = EllipsoidContainer(points)
	plt.plot(ell.x[:,0], ell.x[:,1], 'x', color='b')
	ell.plot(color='b')
	plt.savefig('ellipse.pdf')
	assert ell.contains_points()
	ell.cov[0,1] = 10
	ell.cov[1,0] = ell.cov[0,1]
	ell.plot(color='r')
	plt.savefig('ellipse.pdf')
	assert ell.contains_points()
	plt.clf()
	ell.optimize()
	plt.plot(ell.x[:,0], ell.x[:,1], 'x', color='b')
	ell.plot(color='g')
	plt.savefig('ellipse.pdf')
	

