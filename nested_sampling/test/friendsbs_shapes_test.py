from clustering import clusterdetect
from clustering.neighbors import find_maxdistance, find_rdistance
import matplotlib.pyplot as plt
import numpy
from numpy import log10, log, exp, sin, cos, pi, logical_and
import scipy.spatial, scipy.cluster, scipy.stats
import itertools

ndim = 2
niceplot = True
nresample = 3 if niceplot else 5
speedTest = False

factor = 1
stretch = 1
#stretch = 5 # different scaling in the dimensions
#factor = 3  # number of points
for iresample in range(nresample):
	if niceplot: factor = 2**(iresample)
	numpy.random.seed(4)
	single = numpy.random.normal(0, 0.1, size=200*factor).reshape((100*factor, 2)) + [0.1, 0.7]
	u = numpy.random.uniform(0, 1, size=(10000, 2))
	mask = (u[:,0] - 0.1)**2 + (u[:,1] - 0.7)**2 < 0.3**2
	single = u[mask][:200*factor*0.3**2/(0.3**2 + 0.12**2)]

	single2 = numpy.random.normal(0, 0.04, size=100*factor).reshape((50*factor, 2)) + [0.5, 0.2]
	u = numpy.random.uniform(0, 1, size=(10000, 2))
	mask = (u[:,0] - 0.5)**2 + (u[:,1] - 0.2)**2 < 0.12**2
	single2 = u[mask][:200*factor*0.12**2/(0.3**2 + 0.12**2)]
	
	double = numpy.vstack((single, single2))

	coupled = [numpy.vstack((single2, single2 + [-d, d])) for d in [0.1, 0.2, 0.3, 0.7]]

	sigma = 0.01
	r = 0.93 * sigma
	along = numpy.random.multivariate_normal([0.3, 0.7], [[sigma, r], [r, sigma]], size=100*factor)
	u = numpy.random.uniform(0, 1, size=(10000, 2))
	x, y = u[:,0], u[:,1]
	mask = ((x - y + 0.7 - 0.3)*10)**2 + (x - 0.3)**2 < 0.25**2
	along = u[mask][:100*factor]
	
	ring_radius = 0.2
	phi = numpy.random.uniform(0, 2*pi, size=100*factor)
	jitter = numpy.random.uniform(0, 0.01, size=(100*factor, 2))
	ring = numpy.transpose([0.5 + (ring_radius + jitter[:,0]) * sin(phi), 0.5 + (ring_radius + jitter[:,1]) * cos(phi)])

	along2 = numpy.vstack((along, along + [0.2, -0.2]))

	samples = [single, double, along, along2] + coupled + [ring + jitter]
	if niceplot:
		samples = [double, along, along2] + coupled[:-1] + [ring + jitter]
		t = numpy.linspace(0, 2*pi, 400)
		lines = [[[0.1 + 0.3*sin(t), 0.7 + 0.3*cos(t)],
			  [0.5 + 0.12*sin(t), 0.2 + 0.12*cos(t)]],
			 [[0.3 + 0.25*sin(t), 0.7 + 0.25*(cos(t)/10. + sin(t))]],
			 [[0.3 + 0.25*sin(t), 0.7 + 0.25*(cos(t)/10. + sin(t))],
			  [0.5 + 0.25*sin(t), 0.5 + 0.25*(cos(t)/10. + sin(t))]],
		]
		lines +=[[[0.5 + 0.12*sin(t), 0.2 + 0.12*cos(t)],
			  [0.5 - d + 0.12*sin(t), 0.2 + d + 0.12*cos(t)]] for d in [0.1, 0.2, 0.3, 0.7][:-1]]
		lines +=[[[0.5 + ring_radius*sin(t), 0.5 + ring_radius*cos(t)],
			  [0.5 + (ring_radius+0.01)*sin(t), 0.5 + (ring_radius+0.01)*cos(t)],
			 ]]
		
	if not speedTest:
		if niceplot:
			#plt.figure('resampled', figsize=(10, 5))
			plt.figure('resampled', figsize=(10, 4))
		else:
			plt.figure('resampled', figsize=(2+len(samples)*2, 2+nresample*2))
	for j, data in enumerate(samples):
		data = data[numpy.random.choice(numpy.arange(len(data)), size=50*factor),:]
		data[:,1] *= stretch
		
		if not speedTest:
			plt.subplot(nresample, len(samples), len(samples)*iresample + 1+j)
			#plt.tight_layout()
			#plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
			plt.plot(data[:,0], data[:,1], 'x', ms=2, label='%d samples' % len(data))
			plt.xlim(0, 1)
			plt.ylim(0, 1*stretch)
		
		u = data
		
		#maxdistance = find_maxdistance(u, verbose=True, nbootstraps=15)
		
		maxdistance = find_rdistance(distances = scipy.spatial.distance.cdist(u, u, metric='chebyshev'), 
			verbose=True, nbootstraps=50)
		rdistance = find_rdistance(distances = scipy.spatial.distance.cdist(u, u, metric='euclidean'), 
			verbose=True, nbootstraps=50)

		if speedTest:
			continue
		x = numpy.linspace(0, 1, 100)
		y = numpy.linspace(0, 1*stretch, 100)
		X, Y = numpy.meshgrid(x, y)
		dists = numpy.max([numpy.abs(X.reshape(1,100,100) - u[:,0].reshape(-1,1,1)), 
			 numpy.abs(Y.reshape(1,100,100) - u[:,1].reshape(-1,1,1)) ], axis=0)
		good = dists.min(axis=0) < maxdistance
		plt.contour(X, Y, good, [-0.5, 0.5], colors=['orange'])
		dists = ((X.reshape(1,100,100) - u[:,0].reshape(-1,1,1))**2 + 
			 (Y.reshape(1,100,100) - u[:,1].reshape(-1,1,1))**2)**0.5
		good = dists.min(axis=0) < rdistance
		plt.contour(X, Y, good, [-0.5, 0.5], colors=['red'])
		plt.xticks([]); plt.yticks([])
		#plt.legend(loc='best', frameon=True, size=6)
		
		#plt.text(1.03, -0.03, '%d samples' % len(data), va='top', ha='right',
		#	size=6)
		#plt.text(0.1, 0.1, '%d samples' % len(data), va='center', ha='center',
		#	size=6, rotation=90)
		if j == 0:
			plt.ylabel('%d samples' % len(data), fontsize=10)
		for x, y in lines[j]:
			plt.plot(x, y, '-', color='green', alpha=0.5)

	if not speedTest:
		#plt.gcf().tight_layout()
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.savefig('friendsbs_shapes%s.pdf' % '_nice' if niceplot else '', bbox_inches='tight')
	#plt.close()

