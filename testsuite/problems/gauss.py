"""

Symmetric problems


"""
import numpy
from numpy import pi, exp, log, cos, logaddexp
import scipy.stats, scipy.misc

import scipy.special
def sphere_volume(r, n):
	return pi**(n/2.) / scipy.special.gamma(n/2. + 1) * r**n

def create_problem_gauss(**config):
	ndim = config.get('ndim', 1)
	mu = 0.654321
	sigma = 0.1
	rv = scipy.stats.norm(mu, sigma)
	
	def loglikelihood(x):
		x = numpy.asarray(x)
		#assert numpy.all(x <= 1), (x <= 1, x, '<= 1')
		#assert numpy.all(x >= 0), (x >= 1, x, '>= 0')
		#like = numpy.sum(rv.logpdf(x))
		like = -0.5 * (((x - mu)/sigma)**2 + log(2*pi * sigma**2)).sum()
		#print 'like: %f chi2: %f Z: %f radius: %f' % (
		#	like, 
		#	(((x - mu)/sigma)**2 + log(2*pi * sigma**2)).sum(),
		#	ndim * log(2*pi * sigma**2),
		#	(((x - mu)**2).sum())**0.5
		#	)
		return like
	
	def volume_computation(u=None, x=None, L=None):
		# compute volume within L
		# sphere with the radius
		#radius = (((u - mu)**2).sum())**0.5
		#-2*like = (((x - mu)/sigma)**2 + log(2*pi * sigma**2)).sum()
		chi2 = -2 * L 
		Z = ndim * log(2 * pi * sigma**2) 
		radius = (chi2 - Z)**0.5 * sigma
		#print 'Vlike: %f chi2: %f Z: %f radius: %f' % (L, chi2, Z, radius)
		
		if radius > 1:
			return None
		else:
			return sphere_volume(radius, ndim)
	
	def draw_constrained(x, Lmin):
		L = numpy.sum(rv.logpdf(x[:-1]))
		Llast = Lmin - L
		# pick random x[-1] where constraint fulfilled
		#Llast = -0.5 * ((x - mu)/sigma)**2 - 0.5 * log(2*pi*sigma**2)
		sqwidth = ((-2 * Llast - log(2*pi*sigma**2)) * sigma**2)
		if sqwidth <= 0:
			return []
		else:
			lo = min(0, mu - sqwidth**0.5)
			hi = max(1, mu + sqwidth**0.5)
			#x[ndim - 1] = numpy.random.uniform(lo, hi)
			return [[(ndim - 1, lo, hi)]]
	
	Z_analytic = 0
	config['loglikelihood'] = loglikelihood
	config['draw_constrained'] = draw_constrained
	config['Z_analytic'] = Z_analytic
	config['volume_computation'] = volume_computation
	config['description'] = """A gaussian with mean 0.654321, stdev 0.1.
	In %d dimensions.
	""" % ndim
	return config


def create_problem_gauss_multimodal(**config):
	ndim = config.get('ndim', 2)
	assert ndim >= 2
	
	def loglikelihood(y):
		x = numpy.asarray(y)
		a = ((numpy.abs(x[0:2] - 0.5) - 1./6) * 30)**2
		c = ((x[2:] - 2./3) * 30)**2
		like = -0.5 * (max(a[0], a[1]) + c.sum())
		if like < -300 or numpy.abs(x[0] - 0.5) < 1./6:
			return -300 - ((x - 0.5)**2).sum()
		return like

	def volume_computation(u=None, x=None, L=None):
		# find the closest maximum
		a = ((numpy.abs(x[0:2] - 0.5) - 1./6))**2
		c = ((x[2:] - 2./3))**2
		radius = (max(a[0], a[1]) + c.sum())**0.5
		#print 'volume_computation:', u, radius
		if radius > 2/6. or numpy.abs(x[0] - 0.5) < 1./6:
			return None
		else:
			return sphere_volume(radius, ndim)
		
	#Z_analytic = ndim * (log(30) ) + log(4 / 2.)
	Z_analytic = -(log(30) - log(2*pi)/2)*ndim + log(4/2.)
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['volume_computation'] = volume_computation
	config['description'] = """Combination of Gaussian distributions.
	In %d dimensions. Multi-modal (4).
	""" % ndim
	return config


def create_problem_vargauss(**config):
	ndim = config.get('ndim', 10)
	i = numpy.arange(ndim) + 3.0
	mu = 0.54321
	s = 6./((3.*i)**1.4)
	def loglikelihood(x):
		z = numpy.asarray(x)
		return -0.5 * (((z - mu)/s)**2).sum()
	
	Z_analytic = 0.5 * log(2*pi * s**2).sum()
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """Sequence of independent gaussians which become
	narrower and narrower with the dimension number. Symmetric, %d dimensions.
	""" % ndim
	return config


def create_problem_halfgauss(**config):
	ndim = config.get('ndim', 2)
	assert ndim >= 2
	
	def loglikelihood(y):
		x = numpy.asarray(y)
		c = ((x - 0.5) * 30)**2
		like = -0.5 * c.sum()
		if like < -300 or x[0] < 0.5:
			return -300 - ((x - 0.5)**2).sum()
		return like
	
	#Z_analytic = ndim * (log(30) ) + log(4 / 2.)
	Z_analytic = -(log(30) - log(2*pi)/2)*ndim + log(1/2.)
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """Combination of Gaussian distributions.
	In %d dimensions. Multi-modal (4).
	""" % ndim
	return config


def create_problem_shell(**config):
	ndim = config.get('ndim', 2)
	nmodes = config.get('nmodes', 2)
	width = numpy.array(config.get('width', [0.1/12]*nmodes))
	radius = numpy.array(config.get('radius', [2./12]*nmodes))
	tilt = config.get('tilt', 0)
	centers = numpy.array(config.get('centers', numpy.array([[0.3]*ndim, [0.7]*ndim])))
	
	def loglikelihood(u):
		#x = 12 * numpy.asarray(u) - 6
		l = []
		for sw, sr, sc in zip(width, radius, centers):
			dist = (((sc - u)**2).sum()**0.5 - sr)**2 
			lmode = - dist / (2 * sw**2) - log(2*pi*(sw)**2) / 2
			l.append(lmode)
		l = scipy.misc.logsumexp(l) + (u[0] - 0.5) * tilt
		return l
	def loglikelihood(u):
		x = numpy.asarray(u)
		dist = (((centers - x)**2).sum(axis=1)**0.5 - radius)**2
		lmode = -dist / (2. * width**2) - log(2 * pi * (width)**2) / 2
		l = scipy.misc.logsumexp(lmode) + (x - 0.5).sum() * tilt
		return l

	config['loglikelihood'] = loglikelihood
	
	# analytic solution:
	def shell_vol(r, w):
		# integral along the radius
		mom = scipy.stats.norm.moment(ndim - 1, loc=r, scale=w)
		# integral along the angles is surface of hyper-ball
		# which is volume of one higher dimension x (ndim + 1)
		vol = pi**((ndim)/2.) / scipy.special.gamma((ndim)/2. + 1)
		surf = vol * ndim
		return mom * surf
	Z_analytic = log(sum([shell_vol(sr, sw) for sr, sw in zip(radius, width)]))
	#Z_analytic = log(shell_vol(radius[0], width[0]) * 2)
	# values in arxiv:0809.3437 are off by log(12) in all dimensions
	
	config['Z_analytic'] = Z_analytic
	config['description'] = """A gaussian shell, in %d dimensions.
	Width of shells: %s, radius: %s, centers: %s
	
	<p>From <a href="http://arxiv.org/abs/0704.3704">MultiNest</a> 
	<a href="http://arxiv.org/abs/0809.3437">papers</a></p>
	""" % (ndim, width[0] if (width[0] == width).all() else width, 
		radius[0] if (radius[0] == radius).all() else radius, 
		centers[:,0] if (centers[:,0] == centers.transpose()).all() else centers, 
		)
	return config


def create_problem_eggbox(**config):
	config['ndim'] = 2
	def loglikelihood(u):
		x = 10 * pi * numpy.asarray(u)
		l = (2 + cos(x[0]/2)*cos(x[1]/2))**5
		return l
	# 
	config['Z_analytic'] = 235.88
	config['loglikelihood'] = loglikelihood
	config['description'] = """
	Eggbox problem (2d). Highly multimodal.
	<p>The analytic value has been <a href="http://arxiv.org/abs/0809.3437">computed by fine numerical integration</a></p>
	
	"""
	return config


def create_problem_gen_gauss_sequence(**config):
	ndim = config.get('ndim', 7)
	mu    = 0.5
	# the first distribution is quite broad, so it should be narrower to 
	# be contained in the unit interval
	alpha = numpy.array(([0.01, 0.04, 0.1, 0.1, 0.1, 0.3, 0.3]*10)[:ndim])
	beta =  numpy.array(([0.5,  1,      2,   3, 5,    10, 100]*10)[:ndim])
	norm = log(beta) - log(2*alpha) - scipy.special.gammaln(1./beta)
	def loglikelihood(x): 
		x = numpy.asarray(x)
		return (norm - (numpy.abs(x - mu) / alpha)**beta).sum()
		#return log(beta / (2 * alpha * scipy.special.gamma(1./beta)) * exp(-(numpy.abs(x - mu) / alpha)**beta))
	
	config['Z_analytic'] = 0
	config['loglikelihood'] = loglikelihood
	config['description'] = """
	Generalized gauss functions with beta=%s, center=%s, width=%s. In %d dimensions.
	""" % (beta, mu, alpha, ndim)
	return config


def create_problem_complement_gen_gauss_sequence(**config):
	ndim = config.get('ndim', 7)
	mu    = 0.5
	# the first distribution is quite broad, so it should be narrower to 
	# be contained in the unit interval
	alpha = numpy.array(([0.01, 0.04, 0.1, 0.1, 0.1, 0.3, 0.3]*10)[:ndim])
	beta =  numpy.array(([0.5,  1,      2,   3, 5,    10, 100]*10)[:ndim])
	norm = -log(1 - (2 * alpha * scipy.special.gamma(1./beta))/beta)
	def loglikelihood(x):
		x = numpy.asarray(x)
		return (norm + log(1 - exp(-(numpy.abs(x - mu) / alpha)**beta) + 1e-300)).sum()
	
	config['Z_analytic'] = 0
	config['loglikelihood'] = loglikelihood
	config['description'] = """
	Complement of generalized gauss functions with beta=%s, center=%s, width=%s. In %d dimensions.
	""" % (beta, mu, alpha, ndim)
	return config

def create_problem_triangle(**config):
	ndim = config.get('ndim', 5)
	width = config.get('width', 0.01)
	def loglikelihood(x):
		x = numpy.asarray(x)
		a = numpy.abs(x[0] - 0.5)
		inside = numpy.abs(x[1]-0.5) < width
		like = numpy.where(inside, log(0.5-a), -100 - numpy.abs(x-0.5).sum() )
		return like
	
	# area * height / 2
	config['Z_analytic'] = log(1 * (2 * width) * 0.5 / 2)
	config['loglikelihood'] = loglikelihood
	config['description'] = """
	A triangle in the first dimension. In %d dimensions.
	""" % (ndim)
	return config

def create_problem_triangle_rotated(**config):
	ndim = config.get('ndim', 5)
	width = config.get('width', 0.01)
	def loglikelihood(x):
		x = numpy.asarray(x)
		a = numpy.abs(x[0] - 0.5)
		inside = numpy.logical_and(a < 0.5, numpy.abs((x-0.5).mean()) < 0.01)
		like = numpy.where(inside, log(0.5-a), -100 - numpy.abs(x-0.5).sum())
		return like
	
	# area * height / 2
	config['Z_analytic'] = log(1 * width * ndim / 2)
	config['loglikelihood'] = loglikelihood
	config['description'] = """
	A 45 degree rotated triangle of width=%s. In %d dimensions.
	""" % (width, ndim)
	return config


def create_problem_pyramid(**config):
	ndim = config.get('ndim', 20)
	assert ndim >= 2
	
	def loglikelihood(x):
		x = numpy.asarray(x)
		return -numpy.abs(x-0.5).max()**0.01
	
	Z_analytic = 0
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """Pyramid function. In %d dimensions.
	<p>TODO: the analytic value has not been verified</p>
	""" % ndim
	return config



