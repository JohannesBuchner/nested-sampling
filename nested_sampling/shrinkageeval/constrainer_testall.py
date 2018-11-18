import numpy
from constrainer_test import run_constrainer
from nested_sampling.samplers.rejection import RejectionConstrainer
from nested_sampling.samplers.friends import FriendsConstrainer
from nested_sampling.samplers.mcmc import MCMCConstrainer, GaussProposal, MultiScaleProposal
#from nested_sampling.samplers.affinemcmc import AffineMCMCConstrainer
#from nested_sampling.samplers.galilean import GalileanConstrainer
from nested_sampling.samplers.svm import SVMConstrainer
from nested_sampling.samplers.galilean import MCMCRadFriendsConstrainer
"""
Example test:
"""
def test_bad_mcmc():
	constrainer = MCMCConstrainer(proposer = GaussProposal(adapt=False, scale=1e-5), nmaxsteps=100000)
	N = 40
	for d in 2, 7, 20:
		print('running in %d dimensions' % d)
		run_constrainer(d=d, N=N, constrainer=constrainer, name='rejection')
		plt.savefig('test_constrainer_%d_%s.pdf' % (d, 'rejection'), bbox_inches='tight')
		plt.close()


constrainers = [
	('rejection', lambda : RejectionConstrainer()),
	('multinest', lambda : None),
	('radfriends',  lambda : FriendsConstrainer(rebuild_every=100, radial=True, metric = 'euclidean', jackknife=False, verbose=False)),
	('supfriends',  lambda : FriendsConstrainer(rebuild_every=100, radial=True, metric = 'chebyshev', jackknife=False, verbose=False)),
	#('supfriends1', lambda : FriendsConstrainer(rebuild_every=100, radial=True, metric = 'chebyshev', jackknife=True, verbose=False)),
	('mcmc-gauss-scale-5', lambda : MCMCConstrainer(proposer = GaussProposal(adapt=False, scale=1e-5), nmaxsteps=100000)),
	#('mcmc-gauss-scale0.1', lambda : MCMCConstrainer(proposer = GaussProposal(adapt=False, scale=0.1), nmaxsteps=100000)),
	#('mcmc-gauss-200-adapt', lambda : MCMCConstrainer(nsteps = 200, proposer = GaussProposal(adapt=True, scale=0.1), nmaxsteps=100000)),
	#('mcmc-gauss-50-adapt', lambda : MCMCConstrainer(nsteps = 50, proposer = GaussProposal(adapt=True, scale=0.1), nmaxsteps=100000)),
	#('mcmc-gauss-20-adapt', lambda : MCMCConstrainer(nsteps = 20, proposer = GaussProposal(adapt=True, scale=0.1), nmaxsteps=100000)),
	#('mcmc-gauss-10-adapt', lambda : MCMCConstrainer(nsteps = 10, proposer = GaussProposal(adapt=True, scale=0.1), nmaxsteps=100000)),
	#('mcmc-gauss-200-sivia', lambda : MCMCConstrainer(nsteps = 200, proposer = GaussProposal(adapt='sivia', scale=0.1), nmaxsteps=100000)),
	('mcmc-gauss-50-sivia', lambda : MCMCConstrainer(nsteps = 50, proposer = GaussProposal(adapt='sivia', scale=0.1), nmaxsteps=100000)),
	('mcmc-gauss-20-sivia', lambda : MCMCConstrainer(nsteps = 20, proposer = GaussProposal(adapt='sivia', scale=0.1), nmaxsteps=100000)),
	('mcmc-gauss-10-sivia', lambda : MCMCConstrainer(nsteps = 10, proposer = GaussProposal(adapt='sivia', scale=0.1), nmaxsteps=100000)),
	#('mcmc-radfriends-200', lambda : MCMCRadFriendsConstrainer(proposal_scale = 0.3, nsteps = 200)),
	#('mcmc-radfriends-50',  lambda : MCMCRadFriendsConstrainer(proposal_scale = 0.3, nsteps = 50)),
	#('mcmc-radfriends-20',  lambda : MCMCRadFriendsConstrainer(proposal_scale = 0.3, nsteps = 20)),
	#('mcmc-multiscale-scale3', MCMCConstrainer(proposer = MultiScaleProposal(adapt=False, scale=3), nmaxsteps=100000)),
	#('mcmc-multiscale-scale3-adapt', MCMCConstrainer(proposer = MultiScaleProposal(adapt=True, scale=3), nmaxsteps=100000)),
	#('mcmc-gauss-scale0.1', GalileanConstrainer(nlive_points = nlive_points, ndim = ndim, velocity_scale = velocity_scale)),
	#dict(draw_method='galilean-velocity2', velocity_scale = 0.03),
	#dict(draw_method='galilean-velocity1', velocity_scale = 0.1),
	#dict(draw_method='galilean-velocity3', velocity_scale = 0.001),
	#dict(draw_method='galilean-velocity0', velocity_scale = 0.3),
	#('affinemcmc', lambda : AffineMCMCConstrainer()),
	#('svm', lambda : SVMConstrainer()),
]

def run_constrainers(ds, Ns, constrainers):
	fout = open('constrainertest.tex', 'w')
	print(' %(name)30s  %(d)3s  %(N)3s     D  pvalue     D  pvalue   iter      evals   efficiency' % dict(name='constrainer', d='dim', N='N'))
	print(' %s  ---  ---  ----  ------  ----  ------  -----  ----------  -------------' % ('-'*30))
	for d in ds:
		for name, constrainer in constrainers:
			for N in Ns:
				results = run_constrainer(d, N, constrainer, name=name)
				fout.write('%(name)s & %(d)d & %(pvalue).4f & %(shrinkage_pvalue).4f & %(niter)d & %(total_samples)d & %(efficiency).2f\\%% \\\\ \n' % results)
				fout.flush()
		fout.write('\hline\n')
	fout.close()
def test_all():
	run_constrainers(ds=(2, 7, 20), #(2, 7, 20),
		Ns=[400], constrainers=constrainers)

def test_rejection():
	constrainer = RejectionConstrainer()
	N = 40
	for d in 2, 7, 20:
		print('running in %d dimensions' % d)
		evaluate_constrainer(d=d, N=N, constrainer=constrainer, niter=400)
		plt.savefig('test_constrainer_%d_%s.pdf' % (d, 'rejection'), bbox_inches='tight')
		plt.close()

if __name__ == '__main__':
	import sys
	if len(sys.argv) == 1:
		test_all()
	else:
		sel = int(sys.argv[1])
		i = 0
		ds=(2, 7, 20)
		Ns=[400]
		for name, constrainer in constrainers:
			for d in ds:
				for N in Ns:
					i = i + 1
					if i == sel:
						run_constrainer(d, N, constrainer, name=name)

