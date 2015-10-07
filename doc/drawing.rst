.. _constrained-draw:

Constrained Drawing Functions
==============================

For the problem of drawing uniformly from the prior, given a minimum likelihood,
these are the currently implemented methods:

.. contents::
	:depth: 1
	:local:

Rejection sampling
-------------------
Check the source of this Constrainer for reference, and as a starting point to 
implementing your own method.

.. autoclass:: nested_sampling.samplers.rejection.RejectionConstrainer
	:members:

Support Vector Machines
-------------------------
.. automodule:: nested_sampling.samplers.svm
	:members:

RadFriends/SupFriends
------------------------
.. automodule:: nested_sampling.samplers.friends
	:members:

Markov Chain Monte Carlo
-------------------------
.. automodule:: nested_sampling.samplers.mcmc
	:members: MCMCConstrainer, BaseProposal, GaussProposal, MultiScaleProposal

RadFriends/Galilean and MCMC hybrid methods
--------------------------------------------
.. automodule:: nested_sampling.samplers.galilean
	:members:

Constrained Hamiltonian Monte Carlo
------------------------------------

.. automodule:: nested_sampling.samplers.hamiltonian.hmc
.. automodule:: nested_sampling.samplers.hamiltonian.priors



