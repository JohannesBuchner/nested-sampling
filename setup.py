#!/usr/bin/env python

from distutils.core import setup

long_description = ''
with open('README.rst') as f:
	long_description = f.read()

setup(name='nested_sampling',
	version='0.2',
	author='Johannes Buchner',
	url='https://bitbucket.org/JohannesBuchner/nested-sampling',
	author_email='johannes.buchner.acad@gmx.com',
	description='Nested Sampling testbed and benchmarking',
	long_description=long_description,
	packages=[
		'nested_sampling',
		'nested_sampling.samplers',
		'nested_sampling.clustering',
		'nested_sampling.samplers.svm',
		#'nested_sampling.samplers.hamiltonian',
	],
	)

