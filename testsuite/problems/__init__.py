from __future__ import print_function
"""
Problems known to man
"""
import gauss
import loggamma
#import real.timeseries
import real.bexvar

problems = [
	#gauss.create_problem_gauss(ndim=1, problem_name='gauss1d'),
	gauss.create_problem_gauss(ndim=3, problem_name='gauss3d'),
	gauss.create_problem_gauss(ndim=10, problem_name='gauss10d'),
	#gauss.create_problem_gauss_i(ndim=5, problem_name='gauss_i_5d'),
	#gauss.create_problem_gauss_i(ndim=10, problem_name='gauss_i_10d'),
	#gauss.create_problem_vargauss(ndim=5, problem_name='vargauss5d'),
	#gauss.create_problem_vargauss(ndim=10, problem_name='vargauss10d'),
	#gauss.create_problem_corrgauss(ndim=5, problem_name='corrgauss5d'),
	gauss.create_problem_corrgauss(ndim=10, problem_name='corrgauss10d'),
	#gauss.create_problem_corrgauss(ndim=10, problem_name='corrgauss10d-2', difficulty=2),
	#loggamma.create_problem_bananas(ndim=6, problem_name='bananas6d'),
	#loggamma.create_problem_bananas(ndim=10, problem_name='bananas10d'),
	#loggamma.create_problem_poisson_counts(ndim=3, problem_name='poissoncounts3d'),
	#loggamma.create_problem_poisson_counts(ndim=10, problem_name='poissoncounts10d'),
	#gauss.create_problem_vargauss(ndim=20, problem_name='vargauss20d'),
	#gauss.create_problem_gauss(ndim=50, problem_name='vargauss50d'),
	gauss.create_problem_gauss_i(ndim=2, problem_name='gauss_i_2d'),
	gauss.create_problem_gauss_i(ndim=3, problem_name='gauss_i_3d'),
	gauss.create_problem_gauss_i(ndim=4, problem_name='gauss_i_4d'),
	gauss.create_problem_shell(ndim=2, problem_name='shell2d'),
	#gauss.create_problem_shell(ndim=2, problem_name='shell2dtilt10', tilt=10),
	gauss.create_problem_shell(ndim=10, problem_name='shell10d'),
	#gauss.create_problem_shell(ndim=10, problem_name='shell10d-thin', width=[0.001/12]*2),
	gauss.create_problem_gen_gauss_sequence(ndim=7, problem_name='norm_sequence'),
	gauss.create_problem_complement_gen_gauss_sequence(ndim=7, problem_name='norm_sequence_complement'),
	gauss.create_problem_eggbox(problem_name='eggbox'),
	loggamma.create_problem_rosenbrock(ndim=2, problem_name='rosenbrock2d'),
	loggamma.create_problem_loggamma(ndim=2, problem_name='loggamma2d'),
	loggamma.create_problem_loggamma(ndim=10, problem_name='loggamma10d'),
	#loggamma.create_problem_funnel(ndim=2, problem_name='funnel2d'),
	#loggamma.create_problem_funnel(ndim=2, difficulty=0, problem_name='funnel2d-0'),
	#loggamma.create_problem_funnel(ndim=2, difficulty=1, problem_name='funnel2d-1'),
	#loggamma.create_problem_funnel(ndim=2, difficulty=2, problem_name='funnel2d-2'),
	#loggamma.create_problem_funnel(ndim=5, problem_name='funnel5d'),
	#loggamma.create_problem_funnel(ndim=5, difficulty=0, problem_name='funnel5d-0'),
	#loggamma.create_problem_funnel(ndim=5, difficulty=1, problem_name='funnel5d-1'),
	#loggamma.create_problem_funnel(ndim=5, difficulty=2, problem_name='funnel5d-2'),
	#loggamma.create_problem_funnel(ndim=10, problem_name='funnel10d'),
	#loggamma.create_problem_funnel(ndim=20, problem_name='funnel20d'),
	loggamma.create_problem_ffunnel(ndim=2, problem_name='ffunnel2d-3', difficulty=3),
	loggamma.create_problem_ffunnel(ndim=2, problem_name='ffunnel2d-5', difficulty=5),
	loggamma.create_problem_ffunnel(ndim=5, problem_name='ffunnel5d-3', difficulty=3),
	loggamma.create_problem_ffunnel(ndim=5, problem_name='ffunnel5d-5', difficulty=5),
	loggamma.create_problem_ffunnel(ndim=10, problem_name='ffunnel10d-3', difficulty=3),
	loggamma.create_problem_ffunnel(ndim=10, problem_name='ffunnel10d-5', difficulty=5),
	#real.exvar.create_problem_exvar(ndim=5, problem_name='exvar5d'),
	real.bexvar.create_problem_bexvar(ndim=10, problem_name='bexvar10d', variance=1),
	#real.bexvar.create_problem_bexvar(ndim=10, problem_name='bexvar10d-1', variance=-1),
	#real.exvar.create_problem_exvar(ndim=20, problem_name='exvar20d'),
	#loggamma.create_problem_ffunnel(ndim=20, problem_name='ffunnel20d', difficulty=5),
	loggamma.create_problem_spikeslab(problem_name='spikeslab2d_difficulty3', ndim=2, difficulty=3),
	loggamma.create_problem_spikeslab(problem_name='spikeslab2d_difficulty4', ndim=2, difficulty=4),
	#loggamma.create_problem_spikeslab(problem_name='spikeslab2d_difficulty6', ndim=2, difficulty=6),
	#loggamma.create_problem_spikeslab(problem_name='spikeslab5d_difficulty2', ndim=5, difficulty=2),
	#loggamma.create_problem_spikeslab(problem_name='spikeslab5d_difficulty6', ndim=5, difficulty=6),
	# the slowest last
	loggamma.create_problem_loggammaI_multimodal(ndim=2, problem_name='loggammaI_multimodal2d'),
	#loggamma.create_problem_loggammaI_multimodal(ndim=5, problem_name='loggammaI_multimodal5d'),
	loggamma.create_problem_loggammaI_multimodal(ndim=10, problem_name='loggammaI_multimodal10d'),
	#loggamma.create_problem_loggammaI_multimodal(ndim=20, problem_name='loggammaI_multimodal20d'),
	#loggamma.create_problem_eyes(ndim=5, hardness=1, problem_name='eyes5d'),
	#loggamma.create_problem_eyes(ndim=10, hardness=1, problem_name='eyes10d'),
	#loggamma.create_problem_eyes(ndim=5, hardness=5, problem_name='eyes5d-5'),
	#loggamma.create_problem_eyes(ndim=10, hardness=5, problem_name='eyes10d-5'),
	#real.timeseries.create_problem_RVexoplanet(nplanets=0, problem_name='RVexoplanet-0'),
	#real.timeseries.create_problem_RVexoplanet(nplanets=1, problem_name='RVexoplanet-1'),
	#loggamma.create_problem_eyes(ndim=20, hardness=5, problem_name='eyes20d-5'),
	#real.timeseries.create_problem_RVexoplanet(nplanets=2, problem_name='RVexoplanet-2'),
]


if __name__ == '__main__':
	import time
	import numpy
	N = 10000
	for p in problems:
		like = p['loglikelihood']
		v = numpy.zeros(p['ndim']) + 0.5
		starttime = time.time()
		for i in range(N):
			like(v)
		duration = time.time() - starttime
		
		print('%-25s: %05.03f seconds / 10000 evaluations -- expect %.2f minutes runtime for 2000000' % (p['problem_name'], duration, duration / N * 2000000 / 60))
		
		
