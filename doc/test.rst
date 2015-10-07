.. ref:_test

Benchmark/Test Suite
========================

The benchmark runs the implemented algorithms against a number of test problems,
and ranks them, after evaluating 10 runs with different seeds, by

1. correctness (tolerance of :math:`\Delta Z < 0.1` in accuracy) is fulfilled
2. claimed accuracy does not overestimate actual accuracy
3. number of likelihood evaluations

Following the Benchmark, one may choose the right algorithm for a problem
according to characteristics, such as

* unimodal vs multimodal
* low-dimensional (<7), medium-dimensional (7-20), high-dimensional (30-1000)
* simple peak shape vs peculiar shapes (e.g. bananas, funnels, multiple scales in modes)

Algorithms available
----------------------

All algorithms implemented in the :doc:`python package <index>` are available.
To add your own algorithm, add a if-clause in `testsuite/algorithms/nest.py`.

Additionally, the following algorithms are evaluated in the benchmark:

* Suave (through PyCuba)
* Divonne (through PyCuba)
* Cuhre (through PyCuba)
* Vegas (through PyCuba)
* MultiNest (through PyMultiNest)

Running the Test Suite
-----------------------

To run all the algorithms against all the problems, run::

	$ cd testsuite
	$ mkdir -p output && cd output
	$ PYTHONPATH=../../ python ../testbase.py

This ensures you are using the current source of the nested sampling python 
package.

Parallel execution
---------------------
Parallel execution using multiple threads can be enabled by 
setting the environment variable PARALLEL=1.

Partially running the Test Suite 
----------------------------------

Create a algorithm exclusion file named skip_algorithms, which contains
regular expressions on the names of algorithms to skip. Example::

	cuba.*
	multinest.*
	svm

Create a problem exclusion file named skip_problems, which contains
regular expressions on the names of problems to skip. Example::

	gauss
	tilted
	loggamma[^_]
	.*10.*

As before, run the test suite, but with `partialtestbase.py` ::

	$ cd testsuite
	$ mkdir -p output && cd output
	$ PYTHONPATH=../../ python ../partialtestbase.py


