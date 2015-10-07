Nested Sampling
=======================================

A Pythonic implementation of the Nested Sampling integration algorithm
for Bayesian model comparison and parameter estimation.

This package provides multiple implementations for constrained drawing functions,
and a test suite to evaluate the correctness, accuracy and efficiency of various
implementations.

Currently implemented algorithms
----------------------------------

For drawing uniformly above the likelihood constraint, currently the following
algorithms are implemented:

* rejection sampling
* MCMC with adaptive proposal
* RadFriends/SupFriends
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


