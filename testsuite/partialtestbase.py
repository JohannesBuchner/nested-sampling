import testbase
import problems, algorithms
import re

class REFilter(object):
	def __init__(self, filename):
		lines = open(filename).readlines()
		lines = [l[:-1] for l in lines if not l.startswith('#') and l.strip() != '']
		self.patterns = [re.compile(l) for l in lines]
	def match(self, string):
		return any([p.match(string) for p in self.patterns])

def run_partial():
	"""
	Skips problems specified in skip_problems
	Skips algorithms specified in skip_algorithms
	Each file contains lines of regular expressions
	"""
	problem_filter = REFilter('skip_problems')
	algorithm_filter = REFilter('skip_algorithms')
	
	# get problem
	for i, p in enumerate(problems.problems):
		if problem_filter.match(p['problem_name']):
			print 'skipping', p['problem_name']
			continue
		print 
		print p['problem_name']
		print '=' * len(p['problem_name'])
		# apply algorithm to it, 10 times with different seeds
		algorithms_results = []
		for j, a in enumerate(algorithms.algorithms):
			if algorithm_filter.match(a['algorithm_name']):
				print 'skipping', a['algorithm_name']
				continue
			print '%s (%d of %d)|%s (%d of %d)' % (
				p['problem_name'], i+1, len(problems.problems),
				a['algorithm_name'], j+1, len(algorithms.algorithms)
			)
			algorithms_results.append(testbase.AlgorithmResult(a['algorithm_name'], testbase.run_algorithm(p, a)))
		yield (p, algorithms_results)

if __name__ == '__main__':
	testbase.show_results(run_partial())

