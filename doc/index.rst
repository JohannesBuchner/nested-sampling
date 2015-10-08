Nested Sampling Development Framework & UltraNest
===================================================

About 
------

A Pythonic, modular implementation of the Nested Sampling integration algorithm
for Bayesian model comparison and parameter estimation.

This package consists of three parts:

1. Python package for :ref:`integration using Nested Sampling <nested-sampling>`
2. Python framework for :doc:`testing/benchmarking Nested Sampling <test>`
3. `UltraNest <ultranest/index.html>`_: A C drop-in replacement for MultiNest

.. _nested-sampling:

Pythonic Nested Sampling
------------------------------

Nested Sampling consists of three software components:

1. The :ref:`Nested Sampling integrator <nested-integrator>`.
   It performs the nested sampling integration itself, and computes the evidence.
   This component is very well understood theoretically.
2. The :ref:`Sampler <nested-sampler>`.
   It keeps the current live points sorted, and replaces the least likely point
   in each iteration, using the next component.
   This component is trivial.
3. The :doc:`Constrained Drawing function <drawing>`.
   A method that samples point uniformly from the prior, subject to that their
   likelihood is above a given value. 
   This package provides multiple implementations for constrained drawing functions,
   namely **rejection sampling** (the simplest), **MCMC**, **RadFriends**/**SupFriends**, **Hamiltonian Monte Carlo**.
   See the documentation of :ref:`Implemented Constrained Drawing Functions <constrained-draw>` for more details.
   This component is subject of current research.

The :doc:`Test Suite <test>` can be used to evaluate the correctness, accuracy and efficiency of various
implementations.


Setting up a problem
------------------------------

The problem is specified via a coordinate transform from the unit cube 
(*priortransform*), and the logarithm of the likelihood::

	from numpy import log, pi
        
        # set dimensionality
        ndim = 2
        
        def priortransform(u):
        	# linear transform into [-1 : 1] interval
        	x = u * 2 - 1
        	return x

        def loglikelihood(x):
                a = -0.5 * (((x[0] - 0.5)/0.03)**2)
                b = -0.5 * (((x[1] - 0.5)/0.03)**2)
                return a + b - 0.5 * log(2*pi) * 2 - 0.5 * log(0.03**2) * 2

To integrate, we need the integrator and sampler, and we need to choose a 
constrained drawing function.

Here, we use the simplest setup::

	from nested_integrator import nested_integrator
	from nested_sampler import NestedSampler
	
	# choose constrainer
	# a more specialised constrainer may take more arguments
	from samplers.rejection import RejectionConstrainer as Constrainer
	constrainer = Constrainer()

	print 'preparing sampler...'
	sampler = NestedSampler(nlive_points = 400, # number of live points
		# problem
		priortransform=priortransform,
		loglikelihood=loglikelihood, 
		# constrained drawing function
		draw_constrained = constrainer.draw_constrained, 
		ndim=ndim # dimensionality
		)
	# tell constrainer about sampler so they can interact
	constrainer.sampler = sampler
	
	print 'running sampler...'
	result = nested_integrator(tolerance=0.2, sampler=sampler)
	
	print 'nested sampling (%d samples) logZ = ' % len(result['samples']), result['logZ'], result['logZerr']

Find out more about the implemented :ref:`Constrained Drawing Functions <constrained-draw>`.

Citing this work
-----------------

The correct citation to use is Buchner (2014) published in Statistics and Computing. 

BibTeX::

	@Article{Buchner2014stats,
	  Title                    = {{A statistical test for Nested Sampling algorithms}},
	  Author                   = {Buchner, Johannes},
	  Year                     = {2014},
	  Doi                      = {10.1007/s11222-014-9512-y},
	  Eprint                   = {1407.5459},
	  ISSN                     = {0960-3174},
	  Language                 = {English},
	  Month                    = jul,
	  Pages                    = {1-10},

	  Adsnote                  = {Provided by the SAO/NASA Astrophysics Data System},
	  Adsurl                   = {http://adsabs.harvard.edu/abs/2014arXiv1407.5459B},
	  Archiveprefix            = {arXiv},
	  Journal                  = {Statistics and Computing},
	  Keywords                 = {Nested sampling; MCMC; Bayesian inference; Evidence; Test; Marginal likelihood},
	  Owner                    = {user},
	  Primaryclass             = {stat.CO},
	  Publisher                = {Springer US},
	  Timestamp                = {2014.08.20}
	}


Download & Code repository
------------------------------

The code repository is at
https://github.com/JohannesBuchner/UltraNest

with subfolders 

1) nested_sampling -- Modular framework for nested sampling algorithms and their development
2) testsuite -- Test framework to evaluate the performance and accuracy of algorithms
3) ultranest - A fast C implementation of a mixed RadFriends/MCMC nested sampling algorithm

Installation of the nested_sampling works as usual with 

	python setup.py install

Documentation:
-------------------------------

.. toctree::
	:maxdepth: 1

	integrator
	test
	drawing


Indices and tables
-------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
