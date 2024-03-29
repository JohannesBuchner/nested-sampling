#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except:
    from distutils.core import setup

long_description = None
with open('README.rst') as file:
    long_description = file.read()

setup(
	name='nested_sampling',
	version='0.3',
	url='https://github.com/JohannesBuchner/UltraNest',
	author='Johannes Buchner',
	author_email='johannes.buchner.acad@gmx.com',
	description='Nested Sampling testbed and benchmarking, UltraNest',
	license = "GPLv3",
	long_description=long_description,
	packages=[
		'nested_sampling',
		'nested_sampling.samplers',
		'nested_sampling.clustering',
		'nested_sampling.samplers.svm',
		#'nested_sampling.samplers.hamiltonian',
	],
	setup_requires=['pytest-runner'],
	tests_require=['pytest'],
)

