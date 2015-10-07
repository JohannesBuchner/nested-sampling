ULTRANEST
==========

A nested sampling algorithm.

Uses RadFriends clustered sampling until it becomes inefficient, then
switches to MCMC with adapted step widths.

Compiling and Install
----------------------
You need the progressbar library. Get it from https://github.com/JohannesBuchner/progressbar.
Build libprogressbar.so.

Then you will be able to just run::

	make

See the example.c for a application using UltraNest with a MultiNest-compatible likelihood function.

Building the documentation
---------------------------

Run::

	make doc

And point your browser to doc/api/html/index.html



