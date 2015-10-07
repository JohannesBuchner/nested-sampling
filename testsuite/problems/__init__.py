"""
Problems known to man
"""
import gauss
import loggamma

problems = [
	gauss.create_problem_gauss(ndim=1, problem_name='gauss1d'),
	gauss.create_problem_gauss(ndim=3, problem_name='gauss3d'),
	gauss.create_problem_gauss(ndim=10, problem_name='gauss10d'),
	gauss.create_problem_shell(ndim=2, problem_name='shell2d'),
	#gauss.create_problem_shell(ndim=2, problem_name='shell2dtilt10', tilt=10),
	gauss.create_problem_shell(ndim=10, problem_name='shell10d'),
	gauss.create_problem_gen_gauss_sequence(ndim=7, problem_name='norm_sequence'),
	gauss.create_problem_complement_gen_gauss_sequence(ndim=7, problem_name='norm_sequence_complement'),
	gauss.create_problem_eggbox(problem_name='eggbox'),
	loggamma.create_problem_rosenbrock(ndim=2, problem_name='rosenbrock2d'),
	loggamma.create_problem_loggamma(ndim=2, problem_name='loggamma2d'),
	loggamma.create_problem_loggamma(ndim=2, problem_name='loggamma10d'),
	loggamma.create_problem_funnel(ndim=2, problem_name='funnel2d'),
	loggamma.create_problem_funnel(ndim=5, problem_name='funnel5d'),
	loggamma.create_problem_spikeslab(problem_name='spikeslab2d_difficulty3', ndim=2, difficulty=3),
	loggamma.create_problem_spikeslab(problem_name='spikeslab2d_difficulty4', ndim=2, difficulty=4),
	#loggamma.create_problem_spikeslab(problem_name='spikeslab2d_difficulty6', ndim=2, difficulty=6),
	#loggamma.create_problem_spikeslab(problem_name='spikeslab5d_difficulty2', ndim=5, difficulty=2),
	#loggamma.create_problem_spikeslab(problem_name='spikeslab5d_difficulty6', ndim=5, difficulty=6),
	# the slowest last
	loggamma.create_problem_loggamma_multimodal(ndim=2, problem_name='loggamma_multimodal2d'),
	loggamma.create_problem_loggamma_multimodal(ndim=5, problem_name='loggamma_multimodal5d'),
	loggamma.create_problem_loggamma_multimodal(ndim=10, problem_name='loggamma_multimodal10d'),
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
		
		print '%-25s: %05.03f seconds / 10000 evaluations -- expect %.2f minutes runtime for 2000000' % (p['problem_name'], duration, duration / N * 2000000 / 60)
		
		
