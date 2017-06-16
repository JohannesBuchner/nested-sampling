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
	assert ndim >= 2, ('ndim must be at least 2')
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

def create_problem_loggammaI_multimodal(**config):
	ndim = config.get('ndim', 2)
	assert ndim >= 2, ('ndim must be at least 2')
	
	def loggamma_func(x, loc, scale):
		y = (x - loc) / scale
		return y - exp(y) - log(scale)
	def double_loggamma_func(x, loc1, loc2, scale):
		y = (x - loc1) / scale
		z = (x - loc2) / scale
		return logaddexp(y - exp(y), z - exp(z)) - log(2) - log(scale)
	
	scale = 1./30.
	gauss_norm = - 0.5*log(2*pi*scale**2)
	def gauss_func(x, loc, scale):
		return -0.5*(x-loc)**2/scale**2 + gauss_norm
	def double_gauss_func(x, loc1, loc2, scale):
		return logaddexp(-0.5*(x-loc1)**2/scale**2 + gauss_norm,
			-0.5*(x-loc2)**2/scale**2 + gauss_norm) - log(2)
		
	def loglikelihood(x):
		theta = x
		#return loggamma_func(theta[0], 2./3., 1./30.) + loggamma_func(theta[1], 2./3., 1./30.)
		L1 = double_loggamma_func(theta[0], 2./3, 1./3, 1./30)
		L2 = double_gauss_func(theta[1], 2./3, 1./3, 1./30)
		like = L1 + L2
		for i in range(2, ndim):
			if i <= (ndim+2)/2:
				like += loggamma_func(theta[i], 2./3., 1./30)
			else:
				like += gauss_func(theta[i], 2./3., 1./30)
		return like

	Z_analytic = 0
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """Combination of Gaussian and LogGamma distributions.
	In %d dimensions. Multi-modal (4), heavy-tailed.
	""" % ndim
	return config

def create_problem_bananas(**config):
	ndim = config.get('ndim', 10)
	assert ndim >= 2, ('ndim must be at least 2')
	assert (ndim - 2) % 2 == 0, ('ndim must be even')
	i = numpy.arange((ndim - 2)/2)
	mu = 0.54321
	s = 10**(-(i/2.))
	def loglikelihood(x):
		a = (numpy.asarray(x[2::2]) - 0.7)
		b = (numpy.asarray(x[3::2]) - 0.3) / s
		l = -0.5 * ((exp(a*15) - b*50 + 0.01)**2/0.75**2 + a**2/(0.04)**2)
		return l.sum()
	
	Z_analytic = {10:-27, 6:0}[ndim]
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """Sequence of bananas in dimension pairs. 
	Become narrower and narrower with the dimension number. 
	The first two dimensions are unconstrained.
	Difficult because of peculiar (non-linear, non-gaussian) shape, 
	because the parameter sizes are orders of magnitudes different and 
	because the interesting region tiny.
	Symmetric, %d dimensions. 
	<p>TODO: the analytic value has not been verified</p>
	""" % ndim
	return config




def create_problem_poisson_counts(**config):
	ndim = config.get('ndim', 10)
	assert ndim >= 2, ('ndim must be at least 2')
	counts = numpy.array([100,3,4] + (numpy.arange(2, ndim)).tolist())
	
	def loglikelihood(x):
		z = numpy.asarray(x) * 20
		# our model is:
		r1 = z[0] - 10
		r2 = z[1] - 10
		# first measurement: square of r1 and r2 is roughly 10
		# second measurement: r1 is roughly 3, but fairly uncertain
		# other measurements: various count rates
		predictions = [(r1**2 + r2**2)**0.5 * 100, 
			numpy.abs(r1) * 3,
			numpy.abs(r2) * 3] + z[2:].tolist()
		loglike = scipy.stats.poisson.logpmf(counts, predictions).sum()
		if numpy.isfinite(loglike):
			return loglike
		else:
			return -1e300 * numpy.sum((counts - predictions)**2)
	
	Z_analytic = {10: -38}.get(ndim, 0)
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """Problem based on poisson counts.
	The first two dimensions form bananas/rings similar to the loggamma problem.
	The other dimensions are loggamma functions.
	Assymmetric, %d dimensions. 
	<p>TODO: the analytic value has not been verified</p>
	""" % ndim
	return config




def create_problem_eyes(**config):
	ndim = config.get('ndim', 10)
	hardness = config.get('hardness', 1)
	assert ndim >= 4, ('ndim must be at least 4')
	r = 1
	rerr = 0.1 / hardness
	prod = 0.4
	proderr = 0.5 / hardness**0.5
	i = numpy.arange(1, ndim-3)
	mu = (-2./i + 0.5)
	s = 6./((3.*i)**1.4)
	def loglikelihood(x):
		params = numpy.asarray(x)
		x = params[:4:2] * 4 - 2
		y = params[1:4:2] * 4 - 2
		z = params[4:]

		partring = ((((x**2 + y**2)**0.5 - r)/rerr)**2).sum()
		partx = (((y - prod)/proderr)**2).sum()
		parthigh = (((z - mu)/s)**2).sum()
		chi2 = partring + partx + parthigh

		return -0.5 * chi2
	
	lo = scipy.stats.norm.cdf(0, mu, s)
	hi = scipy.stats.norm.cdf(1, mu, s)
	Z_high = log(hi - lo).sum() + 0.5 * log(2*pi * s**2).sum()
	
	#Z_analytic = {1:{5: -7.4, 10: -15.2, 20: -0}, 
	#	5:{5: -13.3, 10: -21.4, 20: 0}}[hardness][ndim]
	#Z_analytic = {1:-7.35-2.3431, 5:-13.3-2.3431}[hardness]
	#Z_analytic = {1:-3.072, 2: -4.078, 3:-4.697, 4:-5.1467, 5:-5.4978}[hardness] * 2 + Z_high
	#Z_analytic = 0 + Z_high
	Z_analytic = 2*(log(proderr) + log(rerr)) + Z_high
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """Similar to Loggamma_multimodal problem.
	In %d dimensions. Multi-modal (4), peculiar contours (banana to ring-like).
	
	Analytic value verified by fine integration.
	""" % ndim

	"""
	# For calibrating mu, s with i
	for i in range(1,10):
		mu = (-2./i + 0.5)
		x = numpy.linspace(0, 1, 400)
		s = 6./((3.*i)**1.4)
		print mu, s
		#y = ((mu - x)/s)**2
		y = scipy.stats.norm.pdf(x, mu, s)
		plt.plot(x, y)
	plt.show()
	"""
	return config


def create_problem_funnel(**config):
	ndim = config.get('ndim', 2)
	difficulty = config.get('difficulty', 3)
	assert ndim >= 2, ('ndim must be at least 2')
        #xobs = numpy.array([0.499,0.501]*100)[1:ndim]
        xobs = numpy.array([0.5-10**-difficulty,0.5+10**-difficulty]*100)[1:ndim]
	
	def loglikelihood(x):
		x = numpy.asarray(x)
		width = 10**(x[0] * 20 - 10)
		like = -0.5 * (((x[1:]-xobs)/width)**2 + log(2*pi * width**2)).sum()
		#rv = scipy.stats.norm(0.5, width)
		#like = rv.logpdf(x[1:]).sum()
		return like
	
	def draw_constrained(x, Lmin):
		width = 10**(x[0] * 20 - 10)
		mu = xobs
		Llast = Lmin
		sqwidth = (-2 * Llast - log(2*pi*sigma**2)) * sigma**2
		
		if sqwidth <= 0:
			return []
		else:
			lo = min(0, mu - sqwidth**0.5)
			hi = max(1, mu + sqwidth**0.5)
			return [[(i, lo, hi) for i in range(1, ndim)]]
	
	#Z_analytic = {2:-1.66,5:-2.5, 10:-2.5}
	Z_analytic = {
		(2,0):-4.42888,(5,0):-9.48, (10,0):-17,
		(2,1):-0.5176,(5,1):-2.1, (10,1):-3.1,
		(2,2):-0.695676,(5,2):-2.3, (10,2):-3.37,
		(2,3):-0.69567,(5,3):-2.43, (10,3):-3.4,
		(3,0):-6.260,(3,1):-1.62,
		(4,0):-7.9077,(4,1):-2.0,
        }.get((ndim,difficulty), 0)
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """A gaussian with mean 0.5, stdev first parameter.
	In %d dimensions. Unimodal and peculariar shape, heavy contributions off-maximum.
	<p>TODO: the analytic value has not been verified for ndim>2
        </p>
        <!--
        <p>Monte carlo integration for ndim=2: 
        difficulty=0 gives -4.4289 with 300,000,000 MC samples. Fine grid 100,000x100,000 gives -4.428872. Analytic: -4.42896
        difficulty=1 gives -1.1 first, but then goes up to -0.6..-0.4 with 1,000,000,000 MC samples. Fine grid with 10,000x10,000 gives -1.474, from 0.5..1 gives -0.51760, with trapz: -0.5176, analytic: -0.69655
        difficulty=2 gives -1 first, then -0.6+-0.25 with 2000,000,000, Fine grid with 10,000x100,000 gives 0.422175, with trapz: 0.422171, analytic: -0.695676
        difficulty=3 gives -0.9 first, 0.735+-0.01 with 13000,000,000, Fine grid with 10,000x100,000 gives 2.420035, 2.420030, analytic: -0.69567

        ndim=3:
        difficulty=0 gives -6.26047 with 100,000,000 MC samples.
        difficulty=1 gives -1.7 first -1.62 with 1000,000,000 MC samples.
        
        ndim=4:
        difficulty=0 gives -7.9077 with 100,000,000 MC samples.
        difficulty=1 gives -2+-0.05 with 1000,000,000 MC samples.
        
        ndim=5:
        difficulty=0 gives -9.48 with 300,000,000 MC samples.
        difficulty=1 gives -2.1+-0.2 with 1000,000,000 MC samples.
        difficulty=2 gives -2.3 with 200,000,000 MC samples.
        difficulty=3 gives -2.43 with 1000,000,000 MC samples.
        
        ndim=10:
        difficulty=0 gives -17        with 100,000,000 MC samples.
        difficulty=1 gives -3.1+-0.1  with 500,000,000 MC samples.
        difficulty=2 gives -3.37+-0.1 with 350,000,000 MC samples.
        difficulty=3 gives -3.40+-0.05 with 250,000,000 MC samples.
        -->
        <p>%s</p>
	""" % (ndim, str(xobs))
	return config
	
def create_problem_ffunnel(**config):
	ndim = config.get('ndim', 2)
	difficulty = config.get('difficulty', 3)
	assert ndim >= 2, ('ndim must be at least 2')
	
	def loglikelihood(x):
		x = numpy.asarray(x)
		width = 10**(x[0] * 20 - 10)
		like = -0.5 * (((numpy.abs(x[1:]-0.5) + 10**-difficulty)/width)**2 + log(2*pi * width**2)).sum()
		return like
	
	Z_analytic = {
		(2,0):-4.7323,(5,0):-10.463,   (10,0):-19.1014, (20,0): -35.90715,
		(2,1):-3.24778,(5,1):-5.11407, (10,1):-3.1,
		(2,2):-2.4618,(5,2):-3.21154,  (10,2):-3.37,
		(2,3):-2.00452,(5,3):-2.50263, (10,3):-3.42846,     (20,3): -4.15,
		(2,4):-1.687, (5,4): -2.30399, 
		(2,5):-1.4466, (5,5): -2.33, (10,5): -3.3, (20,5): 0,
        }.get((ndim,difficulty), 0)
	config['loglikelihood'] = loglikelihood
	config['Z_analytic'] = Z_analytic
	config['description'] = """A gaussian with mean 0.5, stdev first parameter.
	In %d dimensions. Unimodal and peculariar shape, heavy contributions off-maximum.
	<p>TODO: the analytic value has not been verified for ndim>2
        </p>
	""" % (ndim)
	return config
		
def create_problem_spikeslab(**config):
	ndim = config.get('ndim', 2)
	difficulty = config.get('difficulty', 3)
	assert ndim >= 2, ('ndim must be at least 2')
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
	
	

