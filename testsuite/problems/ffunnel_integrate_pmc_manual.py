from __future__ import print_function
from numpy import log, exp, pi
import numpy
import sys
import pypmc
import numpy as np

from pypmc.tools.indicator import merge_function_with_indicator
from pypmc.density.mixture import create_gaussian_mixture, create_t_mixture
from pypmc.density.student_t import LocalStudentT
from pypmc.sampler.markov_chain import AdaptiveMarkovChain
from pypmc.mix_adapt.variational import GaussianInference
from pypmc.tools import plot_mixture
import matplotlib.pyplot as plt

difficulty = int(sys.argv[1])
ndim = int(sys.argv[2])

def loglikelihood(x):
	width = 10**(x[:,0].reshape(-1,1) * 20 - 10)
	like = -0.5 * (((numpy.abs(x[:,1:] - 0.5) + 10**-difficulty)/width)**2 + log(2*pi * width**2)).sum()
	print(like.shape)
	outside_mask = ~numpy.logical_and(x > 0, x < 1).any(axis=1)
	#if outside_mask.any():
	#	like[outside_mask] = -numpy.inf
	return numpy.where(outside_mask, -numpy.inf, like)


def loglikelihood_single(x):
	width = 10**(x[0] * 20 - 10)
	like = -0.5 * (((numpy.abs(x[1:] - 0.5) + 10**-difficulty)/width)**2 + log(2*pi * width**2)).sum()
	return like

ind = pypmc.tools.indicator.hyperrectangle(numpy.zeros(ndim), numpy.ones(ndim))
log_target = merge_function_with_indicator(loglikelihood_single, ind, -np.inf)

# lets build components manually
K = 10
weights = numpy.ones(K) / K
means = []
covariances = []
for j in range(K):
	mean = np.zeros(ndim) + 0.5
	mean[0] = j*1./K
	means.append(mean)
	sigma = 10**(mean[0]*20 - 10)
	sigma = max(sigma, 10**-difficulty)
	sigma = min(sigma, 3)
	cov = np.eye(ndim) * sigma
	cov[0,0] = (1. / K)
	#print mean, cov
	covariances.append(cov)

mix = create_gaussian_mixture(means, covariances, weights)

N = 40000
sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, mix, prealloc=N)
print('importance sampling ...')
#print('    drawing samples...')
#samples = mix.propose(N, numpy.random)
#print('    computing likelihood ...')
#weights_target = loglikelihood(samples)
#print('    computing weights...')
#weights_proposal = numpy.array([mix.evaluate(sample) for sample in samples])
#weights = exp(weights_target - weights_proposal)
sampler.run(N)
print('importance sampling done')

weights = pypmc.sampler.importance_sampling.combine_weights([samples[:]      for samples in sampler.samples],
                                                            [weights[:][:,0] for weights in sampler.weights],
                                                            [mix]                                 )[:][:,0]
samples = sampler.samples[:]

integral_estimator = weights.sum() / len(weights)
integral_uncertainty_estimator = np.sqrt((weights**2).sum() / len(weights) - integral_estimator**2) / np.sqrt(len(weights)-1)

print('estimated  integral =', integral_estimator, '+-', integral_uncertainty_estimator)
print('estimated ln of integral =', log(integral_estimator))
print('effective sample size', pypmc.tools.convergence.ess(weights))

print('importance sampling done')

vb2 = pypmc.mix_adapt.variational.GaussianInference(sampler.samples[:],
                                                    initial_guess=mix,
                                                    weights=sampler.weights[:][:,0])

# Note: This time we leave "prune" at the default value "1" because we
#       want to keep all components that are expected to contribute
#       with at least one effective sample per importance sampling run.
print('running variational Bayes ...')
vb2.run(1000, rel_tol=1e-8, abs_tol=1e-5, verbose=True)
vb2mix = vb2.make_mixture()

# Now we draw another 10,000 samples with the updated proposal
sampler.proposal = vb2mix
print('running importance sampling ...')
sampler.run(10**4)

# We can combine the samples and weights from the two runs, see reference [Cor+12].
weights = pypmc.sampler.importance_sampling.combine_weights([samples[:]      for samples in sampler.samples],
                                                            [weights[:][:,0] for weights in sampler.weights],
                                                            [mix, vb2mix]                                 ) \
                                                            [:][:,0]
samples = sampler.samples[:]

#weights = pypmc.sampler.importance_sampling.combine_weights([samples[:]      for samples in sampler.samples],
#                                                            [weights[:][:,0] for weights in sampler.weights],
#                                                            [mix]                                 )[:][:,0]
#samples = sampler.samples[:]

integral_estimator = weights.sum() / len(weights)
integral_uncertainty_estimator = np.sqrt((weights**2).sum() / len(weights) - integral_estimator**2) / np.sqrt(len(weights)-1)

print('estimated  integral =', integral_estimator, '+-', integral_uncertainty_estimator)
print('estimated ln of integral =', log(integral_estimator))
print('effective sample size', pypmc.tools.convergence.ess(weights))

plt.figure()
plt.hist2d(samples[:,0], samples[:,1], weights=weights, bins=100, cmap='gray_r')
pypmc.tools.plot_mixture(sampler.proposal, visualize_weights=True, cmap='jet')
plt.colorbar()
plt.clim(0.0, 1.0)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('ln Z = %f' % log(integral_estimator))
plt.savefig('ffunnel_integrate_pmc_%d_%d.pdf' % (ndim, difficulty), bbox_inches='tight')
plt.close()

