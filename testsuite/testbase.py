from __future__ import print_function
#!/usr/bin/env python
# 

import algorithms
import problems
import json
import os
import numpy
import shutil

TARGET_ACCURACY = 0.5

class AlgorithmResult(object):
	def __init__(self, name, results):
		self.name = name
		self.results = results
	def sortkey(self):
		# if failed, misleading, inaccurate, or slow
		res = self.results
		return (res['fail'] or numpy.isnan(res['accuracy']),
			(not (res.get('predictedaccuracy', 1e300) > TARGET_ACCURACY)) and \
			  (not (res['accuracy'] < TARGET_ACCURACY*1.1)) \
			    if 'accuracy' in res and 'predictedaccuracy' in res else 1e300, 
			res.get('accuracy', 1e300) if (res.get('accuracy', 1e300) > TARGET_ACCURACY) else 0,
			int(res.get('neval', 1e300)))
	def __cmp__(self, other):
		for i, (a, b) in enumerate(zip(self.sortkey(), other.sortkey())):
			if i == 3:
				if 2 * a < b:
					return 4
				elif 2 * b < a:
					return -4
				else:
					return 0
			if a != b:
				if a < b:
					return i + 1
				else:
					return -(i + 1)
		return 0



		(namea, resulta) = self.name, self.results
		(nameb, resultb) = other.name, other.results
		
		# first, by correctness: if the algorithm claims to be better than it is
		# to be fair, if the algorithm determines it failed, this is not a failure in accuracy
		fail_a = resulta['fail']
		fail_b = resultb['fail']
		if fail_b and not fail_a:
			return 1
		if fail_a and not fail_b:
			return -1
		bad_a = not fail_a and resulta['accuracy'] > 2*resulta['predictedaccuracy']
		bad_b = not fail_b and resultb['accuracy'] > 2*resultb['predictedaccuracy']
		if bad_b and not bad_a:
			return 1
		if bad_a and not bad_b:
			return -1
		
		if not fail_a and not fail_b:
			# second, by accuracy: the closer to the solution the better
			tol = min(resulta['accuracy'], resultb['accuracy'], TARGET_ACCURACY)
			if (resulta['accuracy'] - resultb['accuracy']) > tol:
				# significant difference
				#print 'significant difference', resulta['predictedaccuracy'], resultb['predictedaccuracy'], tol
				return 2 if resulta['accuracy'] < resultb['accuracy'] else -2
		
		# thirdly, go by evaluations: the fewer the better, but 10% difference does not matter
		assert 'neval' in resulta, resulta
		assert 'neval' in resultb, resultb
		na = resulta['neval']
		nb = resultb['neval']
		if na < nb * 0.9:
			return 3
		if nb < na * 0.9:
			return -3
		
		# no significant difference!
		#print 'no significant difference', namea, nameb
		return 0
        def __lt__(self, other):
        	return self.__cmp__(other) < 0
        def __gt__(self, other):
        	return self.__cmp__(other) > 0
        def __eq__(self, other):
        	return self.__cmp__(other) == 0
        def __le__(self, other):
        	return self.__cmp__(other) <= 0
        def __ge__(self, other):
        	return self.__cmp__(other) >= 0
        def __ne__(self, other):
        	return self.__cmp__(other) != 0

def mkdir(p):
	try:
		os.mkdir(p)
	except OSError as e:
		if e.errno != 17:
			raise e
		pass
	assert os.path.exists(p)

def create_counting_nonnan(function):
	def counting_function(*args, **kwargs):
		counting_function.calls += 1
		l = float(function(*args, **kwargs))
		if numpy.isnan(l) or numpy.isinf(l):
			import sys
			print('FATAL ERROR: in problem %s, function %s' % (p, original_loglikelihood))
			print('FATAL ERROR: invalid likelihood return value at', x, l)
			sys.exit(-1)
		return l
	counting_function.calls = 0
	return counting_function


def run(r, a, seed):
	config = dict(seed=seed)
	config.update(r)
	config.update(a)
	p = config['problem_name']
	mkdir(p)
	p = os.path.join(p, config['algorithm_name'])
	mkdir(p)
	p = os.path.join(p, '%d_' % config['seed'])
	config['output_basename'] = p
	
	# replace likelihood with counting likelihood
	original_loglikelihood = config['loglikelihood']
	#global neval
	#neval = 0
	#def counting_loglikelihood(x):
	#	global neval
	#	neval = neval + 1
	#	l = float(original_loglikelihood(x))
	#	if numpy.isnan(l) or numpy.isinf(l):
	#		import sys
	#		print 'FATAL ERROR: in problem %s, function %s' % (p, original_loglikelihood)
	#		print 'FATAL ERROR: invalid likelihood return value at', x, l
	#		sys.exit(-1)
	#	return l
	config['loglikelihood'] = create_counting_nonnan(original_loglikelihood)
	
	# check if not there already
	resultfile = p + 'results.json'
	if not os.path.exists(resultfile):
		# need to run it
		#try:
		results = config['run'](**config)
		#except AssertionError as e:
		#	print 'FATAL ERROR: in problem %s, algorithm %s' % (p, a['algorithm_name'])
		#	results = dict(fail=True)
		#	print e
		results['neval'] = config['loglikelihood'].calls
		json.dump(results, open(resultfile, 'w'), indent=2)
	
	results = json.load(open(resultfile))
	config.update(results)
	return results

from joblib import Parallel, delayed

current_problem = None
current_algorithm = None
def run_current(seed):
	return run(current_problem, current_algorithm, seed)

def run_algorithm(p, a):
	# evil global variables, but avoids pickling functions
	if a.get('noseed', False):
		# deterministic algorithm, only need to run once
		results = [run(p, a, 0)]*10
	else:
		if os.environ.get('PARALLEL', '0') == '0':
			results = [run(p, a, seed) for seed in range(10)]
		else:
			global current_problem
			global current_algorithm
			current_problem = p
			current_algorithm = a
			results = Parallel(5)(delayed(run_current)(i) for i in range(10))
	
	neval = numpy.mean([result['neval'] for result in results])
	fail = any([result.get('failed', False) for result in results])
	summary = dict(
		neval=neval,
		fail=fail,
		allresults=results
	)
	try:
		value = numpy.mean([result['Z_computed'] for result in results])
		accuracy = numpy.mean([(result['Z_computed'] - p['Z_analytic'])**2 for result in results])**0.5
		predictedaccuracy = numpy.mean([result['Z_computed_err'] for result in results])
		summary.update(dict(
			value=value,
			accuracy=accuracy, 
			predictedaccuracy=predictedaccuracy))
	except KeyError as e:
		pass
	return summary

def run_all():
	# get problem
	for i, p in enumerate(problems.problems):
		print()
		print( p['problem_name'])
		print('=' * len(p['problem_name']))
		# apply algorithm to it, 10 times with different seeds
		algorithms_results = []
		for j, a in enumerate(algorithms.algorithms):
			print('%s (%d of %d)|%s (%d of %d)' % (
				p['problem_name'], i+1, len(problems.problems),
				a['algorithm_name'], j+1, len(algorithms.algorithms)
			))
			algorithms_results.append(AlgorithmResult(a['algorithm_name'], run_algorithm(p, a)))
		yield (p, algorithms_results)

def latexname(name):
	for i in range(10):
		name = name.replace(str(i), ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'][i])
	return name.replace('_', '')

def show_results(results):
	fout = open('progress.html', 'w')
	latexout = open('progress.tex', 'w')
	header = "\n</head>\n<body>\n"
	try:
		with open('page_extraheader.html') as f:
			header = f.read()
	except IOError:
		pass
	fout.write("""
<html>
<head>
	<title>Nested Sampling Algorithm comparison</title>
%s""" % header)
	for problem, algs in results:
		fout.write("""
<h2><a name="%(problem_name)s"">%(problem_name)s</a> 
<a href="#%(problem_name)s" class="paralink">&para;</a></h2>
<div class="problem"><p>%(description)s
<p class="analytic">Analytic solution: %(Z_analytic).4f</div>
<table>
<thead>
<tr><th>Algorithm</th>
<th>Rank</th>
<th>Z Result</th>
<th>Accuracy</th>
<th>Claimed accuracy</th>
<th>evaluations</th>
</thead>
<tbody>
""" % problem)
		latexout.write(r"""\newcommand{\problem%s}{
%%\begin{tabular}{| l | l | r | r | r | }
%% Algorithm & Rank & $\ln~Z$ result & A & C & evaluations \\
%% \hline
""" % latexname(problem['problem_name']))

		# rank algorithms
		#algs_ranked = sorted(algs, reverse=False)
		algs_ranked = sorted(algs, key=AlgorithmResult.sortkey)
		
		rank = 0
		rank_last = None
		for alg in algs_ranked:
			algname, results = alg.name, alg.results
			if rank_last is None:
				rank += 1
				rank_text = "%d" % rank
				rank_reason = 100
			else:
				rank_reason = -alg.__cmp__(rank_last)
				#assert rank_reason >= 0, rank_reason
				if rank_reason != 0:
					rank += 1
					#reasons = ['same', 'correctness', 'accuracy', 'speed']
					reasons = ['same', 'failed', 'misleading', 'accuracy', 'speed']
					rank_text = "%d (%s)" % (rank, reasons[abs(rank_reason)])
				else:
					rank_text = "%d =" % rank
			if rank_reason != 0:
				rank_last = alg
			fout.write("""<tr><th>%s%s</th><td>%s</td>""" % (algname, '***' if results['fail'] else '', rank_text))
			latexout.write("%s & %s & " % (algname, rank_text))
			if results['fail']:
				fout.write("""
					<td class="result-wrong">(failure)</td>
					<td class="result-wrong">(failure)</td>
					<td class="result-wrong">(failure)</td>
					<td>%d</td>
				""" % results['neval'])
				latexout.write(r"(failure) & (failure) & (failure) & %d " % results['neval'])
			else:
				result_class = 'ok' if abs(results['value'] - problem['Z_analytic']) < TARGET_ACCURACY else 'wrong'
				accuracy_class = 'ok' if abs(results['accuracy']) < TARGET_ACCURACY else 'wrong'
				claim_class = 'ok' if results['accuracy'] < 2 * results['predictedaccuracy'] else 'wrong'
				fmt = dict(result_class=result_class, accuracy_class=accuracy_class, claim_class=claim_class)
				fmt.update(results)
				fout.write("""
					<td class="result-%(result_class)s">%(value).4f</td>
					<td class="result-%(accuracy_class)s">%(accuracy).4f</td>
					<td class="result-%(claim_class)s">%(predictedaccuracy).4f</td>
					<td>%(neval)d</td>
				""" % fmt)
				latexout.write(r"%(value).4f & %(accuracy).4f & %(predictedaccuracy).4f & %(neval)d " % fmt)
			fout.write("""</tr>\n""")
			latexout.write("\\\\\n")

		fout.write("""</tbody>\n</table>\n""")
		latexout.write("""%\\hline\n}\n\n""")
		fout.flush()
		latexout.flush()
	fout.write("""\n</html>""")
	fout.close()
	shutil.copyfile('progress.html', 'index.html')
	shutil.copyfile('progress.tex', 'index.tex')


if __name__ == '__main__':
	problems.problems
	show_results(run_all())
	

