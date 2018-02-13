import numpy
import os
from ctypes import *
import scipy.spatial
from numpy.ctypeslib import ndpointer

if int(os.environ.get('OMP_NUM_THREADS', '1')) > 1:
        libname = 'neighbors-parallel.so'
else:
        libname = 'neighbors.so'

lib = cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), libname))
lib.most_distant_nearest_neighbor.argtypes = [
        ndpointer(dtype=numpy.float64, ndim=2, flags='C_CONTIGUOUS'), 
        c_int, 
        c_int, 
        ]
lib.most_distant_nearest_neighbor.restype = c_double

def most_distant_nearest_neighbor(xx):
        i, m = xx.shape
        r = lib.most_distant_nearest_neighbor(xx, i, m)
        return r

def nearest_rdistance_guess(u, metric='euclidean'):
	n = len(u)
	distances = scipy.spatial.distance.cdist(u, u, metric=metric)
	#print distances
	numpy.fill_diagonal(distances, 1e300)
	nearest_neighbor_distance = numpy.min(distances, axis = 1)
	#print nearest_neighbor_distance
	rdistance = numpy.max(nearest_neighbor_distance)
	#print 'distance to nearest:', rdistance, nearest_neighbor_distance
	return rdistance

if __name__ == '__main__':
	for i in 1, 2, 3, 4, 5:
		print('==== INPUT ====')
		print('SEED=%d' % i)
		numpy.random.seed(i)
		xx = numpy.random.uniform(size=(300, 2))
		#xx[0,:] = 0.
		print(xx)
		print('==== PYTHON ====')
		rpy = nearest_rdistance_guess(xx)
		print(rpy)
		print('==== C ====')
		rc = most_distant_nearest_neighbor(xx)
		print(rc)
		assert rc == rpy, (rc, rpy)
	print('speed test...')
	xx = numpy.random.uniform(size=(30000, 2))
	print(most_distant_nearest_neighbor(xx))
	print(most_distant_nearest_neighbor(xx))
	print(most_distant_nearest_neighbor(xx))
	
