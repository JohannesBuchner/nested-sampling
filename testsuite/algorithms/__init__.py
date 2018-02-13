"""
Algorithms known to man
"""
from . import multinest
from . import cuba
from . import nest
from . import runnestle

algorithms = \
	multinest.configs + \
	runnestle.configs + \
	cuba.configs + \
	nest.configs


