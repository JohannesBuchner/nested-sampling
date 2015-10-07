"""
Asymmetric problems

"""
import numpy
from numpy import pi, exp, log, cos, logaddexp
import scipy.stats

def create_problem_rosenbrock(**config):
	ndim = config.get('ndim', 2)
	def loglikelihood(u):
		l = 0
		x = 10 * numpy.asarray(u) - 5
		for i in range(ndim - 1):
			l -= (1 - x[i])**2 + 100 * (x[i+1] - x[i]**2)**2
		return l
	# TODO: got this from multinest results!
	config['Z_analytic'] = -5.8 if ndim == 2 else -1000
	config['loglikelihood'] = loglikelihood
	config['description'] = """
	Rosenbrocks valley. In %d dimensions.
	<p>TODO: the analytic value has not been verified</p>
	""" % ndim
	return config

def create_problem_loggamma(**config):
	ndim = config.get('ndim', 1)
	rv = scipy.stats.loggamma(1, loc=0.5, scale=1./30)
	
	def loglikelihood(x):
		like = numpy.sum(rv.logpdf(x))
		# avoid -inf
		if like < -300:
			return -300 - ((numpy.asarray(x) - 0.5)**2).sum()
		return like

	Z_analytic = 0
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """LogGamma distribution at 0.5, scale 1/30, shape 1.
	In %d dimensions. Monomodal, but heavy-tailed.
	""" % ndim
	return config

def create_problem_loggamma_multimodal(**config):
	ndim = config.get('ndim', 2)
	assert ndim >= 2
	rv1a = scipy.stats.loggamma(1, loc=2./3, scale=1./30)
	rv1b = scipy.stats.loggamma(1, loc=1./3, scale=1./30)
	rv2a = scipy.stats.norm(2./3, 1./30)
	rv2b = scipy.stats.norm(1./3, 1./30)
	rv_rest = []
	for i in range(2, ndim):
		if i <= (ndim+2)/2:
			rv = scipy.stats.loggamma(1, loc=2./3., scale=1./30)
		else:
			rv = scipy.stats.norm(2./3, 1./30)
		rv_rest.append(rv)
	rv_rest_enum = [(i+2, rv) for i, rv in enumerate(rv_rest)]
	
	def loglikelihood(x):
		theta = x
		L1 = log(0.5 * rv1a.pdf(theta[0]) + 0.5 * rv1b.pdf(theta[0]))
		L2 = log(0.5 * rv2a.pdf(theta[1]) + 0.5 * rv2b.pdf(theta[1]))
		like = L1 + L2 + sum([rv.logpdf(t) for rv, t in zip(rv_rest, theta[2:])])
		#like = L1 + L2 + sum([rv_rest[i].logpdf(t) for i, t in enumerate(theta[2:])])
		#like = L1 + L2
		#for i in range(ndim-2):
		#	like += rv_rest[i].logpdf(theta[i+2])
		#for i, rv in rv_rest_enum:
		#	like += rv.logpdf(theta[i])
		#like = L1 + L2 + sum((rv.logpdf(theta[i]) for i, rv in rv_rest_enum))
		# avoid -inf
		if like < -300:
			return -300 - ((numpy.asarray(x) - 0.5)**2).sum()
		return like

	Z_analytic = 0 # -ndim * log(60)
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """Combination of Gaussian and LogGamma distributions.
	In %d dimensions. Multi-modal (4), heavy-tailed.
	""" % ndim
	return config


def create_problem_funnel(**config):
	ndim = config.get('ndim', 2)
	assert ndim >= 2
	
	def loglikelihood(x):
		x = numpy.asarray(x)
		width = 10**(x[0] * 20 - 10)
		like = -0.5 * (((x[1:]-0.5)/width)**2 + log(2*pi * width**2)).sum()
		#rv = scipy.stats.norm(0.5, width)
		#like = rv.logpdf(x[1:]).sum()
		return like
	
	def draw_constrained(x, Lmin):
		width = 10**(x[0] * 20 - 10)
		mu = 0.5
		Llast = Lmin
		sqwidth = (-2 * Llast - log(2*pi*sigma**2)) * sigma**2
		
		if sqwidth <= 0:
			return []
		else:
			lo = min(0, mu - sqwidth**0.5)
			hi = max(1, mu + sqwidth**0.5)
			return [[(i, lo, hi) for i in range(1, ndim)]]
	
	Z_analytic = -1.66 if ndim == 2 else -2.5 # TODO
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """A gaussian with mean 0.5, stdev first parameter.
	In %d dimensions. Unimodal and peculariar shape, heavy contributions off-maximum.
	<p>TODO: the analytic value has not been verified</p>
	""" % ndim
	return config
	
	
def create_problem_spikeslab(**config):
	ndim = config.get('ndim', 2)
	difficulty = config.get('difficulty', 3)
	assert ndim >= 2
	sigma1 = 0.1
	mu1 = 0.5
	sigma2 = 10**-difficulty
	mu2 = 0.5 + 0.031
	
	
	def loglikelihood(x):
		x = numpy.asarray(x)
		L1 = -0.5 * (((x - mu1)/sigma1)**2 + log(2*pi * sigma1**2)).sum()
		L2 = -0.5 * (((x - mu2)/sigma2)**2 + log(2*pi * sigma2**2)).sum()
		like = logaddexp(L1, L2 + log(100))
		return like
	
	Z_analytic = log(1+100)
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """A gaussian with mean %s, stdev %s (the slab). On top of 
	this, with 100x more evidence, a gaussian with mean %s, stdev %s (the spike). 
	In %d dimensions. Unimodal and peculariar shape -- the small peak is difficult to find.
	""" % (mu1, sigma1, mu2, sigma2, ndim)
	return config
	
	

