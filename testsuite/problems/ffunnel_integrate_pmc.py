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
	width = 10**(x[0].reshape(-1,1) * 20 - 10)
	like = -0.5 * (((numpy.abs(x[1:] - 0.5) + 10**-difficulty)/width)**2 + log(2*pi * width**2)).sum()
	return like

ind = pypmc.tools.indicator.hyperrectangle(numpy.zeros(ndim), numpy.ones(ndim))
log_target = merge_function_with_indicator(loglikelihood, ind, -np.inf)

mcs = []
for j in range(10):
	print('running MCMC chain %d ...' % j)
	start = np.zeros(ndim) + 0.5
	start[0] = np.random.uniform(0, 1)
	sigma = 10**(start[0]*20 - 10)
	if sigma > 0.5:
		sigma = 0.5
	prop = LocalStudentT(0.2 * sigma, 1)
	mc = pypmc.sampler.markov_chain.AdaptiveMarkovChain(log_target, prop, start)
	mcs.append(mc)
	
	mc.run(500)
	mc.clear()
	for i in range(20):
		mc.run(1000)
		mc.adapt()

mc_samples_sorted_by_chain = [mc.samples[:] for mc in mcs]
mc_samples = np.vstack(mc_samples_sorted_by_chain)

means = np.zeros((len(mcs), ndim))
variances = np.zeros_like(means)

for i,mc in enumerate(mc_samples_sorted_by_chain):
	means[i] = mc.mean(axis=0)
	variances[i] = mc.var(axis=0)

# create a initial gaussian mixture based on the MCMC samples
long_patches = pypmc.mix_adapt.r_value.make_r_gaussmix(mc_samples_sorted_by_chain, K_g=10)
vb = pypmc.mix_adapt.variational.GaussianInference(mc_samples[::100], 
	initial_guess=long_patches, W0=np.eye(ndim)*1e10)

vb_prune = 0.5 * len(vb.data) / vb.K

print('running variational Bayes ...')
vb.run(1000, rel_tol=1e-8, abs_tol=1e-5, prune=vb_prune, verbose=True)
print('running variational Bayes ... done')

vbmix = vb.make_mixture()

print('importance sampling ...')
sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, vbmix)
sampler.run(10000)

print('creating another variational Bayes...')
prior_for_proposal_update = vb.posterior2prior()
prior_for_proposal_update.pop('alpha0')
vb2 = pypmc.mix_adapt.variational.GaussianInference(sampler.samples[:],
                                                    initial_guess=vbmix,
                                                    weights=sampler.weights[:][:,0],
                                                    **prior_for_proposal_update)

print('running variational Bayes ...')
vb2.run(1000, rel_tol=1e-8, abs_tol=1e-5, verbose=True)
vb2mix = vb2.make_mixture()

sampler.proposal = vb2mix
print('running importance sampling ...')
sampler.run(10**5)

weights = pypmc.sampler.importance_sampling.combine_weights([samples[:]      for samples in sampler.samples],
                                                            [weights[:][:,0] for weights in sampler.weights],
                                                            [vbmix, vb2mix]                                 ) \
                                                            [:][:,0]
samples = sampler.samples[:]

integral_estimator = weights.sum() / len(weights)
integral_uncertainty_estimator = np.sqrt((weights**2).sum() / len(weights) - integral_estimator**2) / np.sqrt(len(weights)-1)

print('analytical integral = 1')
print('estimated  integral =', integral_estimator, '+-', integral_uncertainty_estimator)
print('estimated ln of integral =', log(integral_estimator))
print('effective sample size', pypmc.tools.convergence.ess(weights))

plt.figure()
plt.hist2d(samples[:,0], samples[:,1], weights=weights, bins=100, cmap='gray_r')
pypmc.tools.plot_mixture(sampler.proposal, visualize_weights=True, cmap='jet')
plt.colorbar()
plt.clim(0.0, 1.0)
plt.title('ln Z = %f' % log(integral_estimator))
plt.savefig('funnel_integrate_pmc_%d_%d.pdf' % (ndim, difficulty), bbox_inches='tight')
plt.close()

"""
data = mc.samples[:][::100]
K = 20
vb = GaussianInference(data, components=K, alpha=10*np.ones(K), nu=3*np.ones(K))

initial_mix = vb.make_mixture()
vb.run(prune=0.5*len(data) / K, verbose=True)
mix = vb.make_mixture()
"""


