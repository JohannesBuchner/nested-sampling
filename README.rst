Nested Sampling Development Framework & UltraNest
===================================================

A Pythonic implementation of the Nested Sampling integration algorithm
for Bayesian model comparison and parameter estimation.

This package provides multiple implementations for constrained drawing functions,
and a test suite to evaluate the correctness, accuracy and efficiency of various
implementations.

The source code consists of three components:

1) A Modular framework for nested sampling algorithms (nested_sampling) and their development
2) Test framework to evaluate the performance and accuracy of algorithms (testsuite)
3) UltraNest - A fast C implementation of a mixed RadFriends/MCMC nested sampling algorithm

Currently implemented algorithms
----------------------------------

For drawing uniformly above the likelihood constraint, currently the following
algorithms are implemented:

* naive rejection sampling
* MCMC with adaptive proposal
* RadFriends/SupFriends (see Buchner, 2014)
* SVM-based constrainer
* Galilean (in progress)
* Hamiltonian (in progress)

Test Suite
----------------------------------

For testing the correctness and efficiency of algorithms, multiple algorithms
can be run against a set of test problems.

Additionally to nested sampling implemented here, the following algorithms are evaluated in the benchmark:

* MultiNest (through PyMultiNest)
* Cuba-based algorithms (through PyCuba)

  * Suave 
  * Divonne (through PyCuba)
  * Cuhre (through PyCuba)
  * Vegas (through PyCuba)


Getting started
----------------

Have a look at the file nested_sampling/test/simplenested_test.py
to build your own combination of nested sampling algorithm components.

You can run the entire test suite using the testsuite/testbase.py
but to only run a part of it create skip_algorithms and skip_problems files and 
run testsuite/partialtestbase.py

The documentation is available at https://johannesbuchner.github.io/UltraNest/
The source code repository is available at https://github.com/JohannesBuchner/UltraNest/

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


License
------------------

This work is open source software licensed under GPLv3 Affero (see LICENSE file). 
If you are interested in a different license, please contact me. 

