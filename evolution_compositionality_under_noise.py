import itertools
import numpy as np
import random
from scipy.special import logsumexp
from copy import deepcopy
import matplotlib.pyplot as plt


# FROM SIMLANG LAB 21:
languages = [[('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')]]
types = [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]
priors = [-0.9178860550328204, -10.749415928290118, -10.749415928290118, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -12.460704095246543, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -12.460704095246543, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -2.304180416152711, -11.272664072079987, -10.749415928290118, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -10.749415928290118, -11.272664072079987, -2.304180416152711, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -12.460704095246543, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -12.460704095246543, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -10.749415928290118, -10.749415928290118, -0.9178860550328204]


# MY OWN CODE:

# First some parameters:
meanings = ['02', '03', '12', '13']  # all possible meanings
forms_without_noise = ['aa', 'ab', 'ba', 'bb']  # all possible forms, excluding their possible 'noisy variants'
noisy_forms = ['a_', 'b_', '_a', '_b']  # all possible noisy variants of the forms above
all_forms_including_noisy_variants = forms_without_noise+noisy_forms # all possible forms, including both complete forms and noisy variants
error = 0.05  # the probability of making a production error
gamma = 2  # parameter that determines strength of ambiguity penalty (Kirby et al. used gamma = 2 for communication
            # condition and gamma = 0 for condition without communication)
turnover = True  # determines whether new individuals enter the population or not
noise = False  # parameter that determines whether environmental noise is on or off
noise_prob = 0.1  # the probability of environmental noise masking part of an utterance



# Some functions to create and classify all possible languages:
def create_all_possible_languages(meanings, forms):
    """Creates all possible languages

    :param meanings: list of strings corresponding to all possible meanings
    :type meanings: list
    :param forms: list of strings corresponding to all possible forms_without_noisy_variants
    :type forms: list
    :returns: list of tuples which represent languages, where each tuple consists of forms_without_noisy_variants and has length len(meanings)
    :rtype: list
    """
    all_possible_languages = list(
        itertools.product(forms, repeat=len(meanings)))
    return all_possible_languages


def classify_language(lang, forms, meanings):
    """
    Classify one particular language as either 'degenerate' (0), 'holistic' (1), 'other' (2)
    or 'compositional' (3) (Kirby et al., 2015)

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same index in meanings
    :type lang: tuple
    :param forms: list of strings corresponding to all possible forms_without_noisy_variants
    :type forms: list
    :returns: integer corresponding to category that language belongs to:
    0 = degenerate, 1 = holistic, 2 = other, 3 = compositional (here I'm following the
    numbering used in SimLang lab 21)
    :rtype: int
    """
    # TODO: See if I can modify this function so that it can deal with any number of forms_without_noisy_variants and meanings.
    class_degenerate = 0
    class_holistic = 1
    class_other = 2
    class_compositional = 3
    # First check whether some conditions are met, bc this function hasn't been coded up in the most general way yet:
    if len(forms) != 4:
        raise ValueError(
            "This function only works for a world in which there are 4 possible forms_without_noisy_variants"
        )
    if len(forms[0]) != 2:
        raise ValueError(
            "This function only works when each form consists of 2 elements")
    if len(lang) != len(meanings):
        raise ValueError("Lang should have same length as meanings")

    # lang is degenerate if it uses the same form for every meaning:
    if lang[0] == lang[1] and lang[1] == lang[2] and lang[2] == lang[3]:
        return class_degenerate

    # lang is compositional if it makes use of all possible forms_without_noisy_variants, *and* each form element maps to the same meaning
    # element for each form:
    elif forms[0] in lang and forms[1] in lang and forms[2] in lang and forms[
        3] in lang and lang[0][0] == lang[1][0] and lang[2][0] == lang[3][0] and lang[0][
        1] == lang[2][1] and lang[1][1] == lang[3][1]:
        return class_compositional

    # lang is holistic if it is *not* compositional, but *does* make use of all possible forms_without_noisy_variants:
    elif forms[0] in lang and forms[1] in lang and forms[2] in lang and forms[
        3] in lang:
        return class_holistic

    # In all other cases, a language belongs to the 'other' category:
    else:
        return class_other


# Let's try out our create_all_possible_languages() function:
all_possible_languages = create_all_possible_languages(meanings, forms_without_noise)
# print("all_possible_languages are:")
# print(all_possible_languages)
print("number of possible languages is:")
print(len(all_possible_languages))

# Let's test our classify_language() function using some example languages from the Kirby et al. (2015) paper:
degenerate_lang = ('aa', 'aa', 'aa', 'aa')
print('')
print("degenerate_lang is:")
print(degenerate_lang)
class_degenerate_lang = classify_language(degenerate_lang, forms_without_noise, meanings)
print("class_degenerate_lang is:")
print(class_degenerate_lang)

holistic_lang = ('aa', 'ab', 'bb', 'ba')
print('')
print("holistic_lang is:")
print(holistic_lang)
class_holistic_lang = classify_language(holistic_lang, forms_without_noise, meanings)
print("class_holistic_lang is:")
print(class_holistic_lang)

other_lang = ('aa', 'aa', 'aa', 'ab')
print('')
print("other_lang is:")
print(other_lang)
class_other_lang = classify_language(other_lang, forms_without_noise, meanings)
print("class_other_lang is:")
print(class_other_lang)

compositional_lang = ('aa', 'ab', 'ba', 'bb')
print('')
print("compositional_lang is:")
print(compositional_lang)
class_compositional_lang = classify_language(compositional_lang, forms_without_noise,
                                             meanings)
print("class_compositional_lang is:")
print(class_compositional_lang)


def classify_all_languages(language_list):
    """
    Classify all languages as either 'degenerate' (0), 'holistic' (1), 'other' (2) or 'compositional' (3) (Kirby et al., 2015)

    :param language_list: list of all languages
    :type language_list: list
    :returns: 1D numpy array containing integer corresponding to category of corresponding
    language index: 0 = degenerate, 1 = holistic, 2 = other, 3 = compositional
    (here I'm following the numbering used in SimLang lab 21)
    :rtype: 1D numpy array
    """
    class_per_lang = np.zeros(len(language_list))
    for l in range(len(language_list)):
        class_per_lang[l] = classify_language(language_list[l], forms_without_noise, meanings)
    return class_per_lang


# Let's check whether the functions in this cell work correctly by comparing the number of languages of each type we
# get with the SimLang lab 21:

types = np.array(types)
no_of_each_type = np.bincount(types)
print('')
print("no_of_each_type ACCORDING TO SIMLANG CODE is:")
print(no_of_each_type)

class_per_lang = classify_all_languages(all_possible_languages)
print('')
print('')
# print("class_per_lang is:")
# print(class_per_lang)
no_of_each_class = np.bincount(class_per_lang.astype(int))
print('')
print("no_of_each_class ACCORDING TO MY CODE is:")
print(no_of_each_class)


# Hmmm, that gives us slightly different numbers! Is that caused by a problem in my
# create_all_languages() function, or in my classify_lang() function?
# To find out, let's compare my list of all languages to that from SimLang lab 21:

# First, we need to change the way we represent the list of all languages to match
# that of lab 21:

def transform_all_languages_to_simlang_format(language_list):
    """
    Takes a list of languages as represented by me (with only the forms_without_noisy_variants listed
    for each language, assuming the meaning for each form is specified by the
    form's index), and turning it into a list of languages as represented in
    SimLang lab 21 (which in turn is based on Kirby et al., 2015), in which a
    <meaning, form> pair forms_without_noisy_variants a tuple, and four of those tuples in a list form
    a language

    :param language_list: list of all languages
    :type language_list: list
    :returns: list of the input languages in the format of SimLang lab 21
    :rtype: list
    """
    all_langs_as_in_simlang = []
    for l in range(len(all_possible_languages)):
        lang_as_in_simlang = [(meanings[x], all_possible_languages[l][x]) for x in range(len(meanings))]
        all_langs_as_in_simlang.append(lang_as_in_simlang)
    return all_langs_as_in_simlang


all_langs_as_in_simlang = transform_all_languages_to_simlang_format(all_possible_languages)
print('')
print('')
# print("all_langs_as_in_simlang is:")
# print(all_langs_as_in_simlang)
print("len(all_langs_as_in_simlang) is:")
print(len(all_langs_as_in_simlang))
print("len(all_langs_as_in_simlang[0]) is:")
print(len(all_langs_as_in_simlang[0]))
print("len(all_langs_as_in_simlang[0][0]) is:")
print(len(all_langs_as_in_simlang[0][0]))


def check_all_lang_lists_against_each_other(language_list_a, language_list_b):
    """
    Takes two lists of languages of the same length and format, and checks for each languages in language_list_a,
    whether it is also present in language_list_b.

    :param language_list_a: list of languages represented as in the SimLang lab 21 code, where each language is a list
    of 4 tuples, where each tuple consists of a meaning and its corresponding form.
    :param language_list_b: list of languages of same format as language_list_b
    :return: a list of binary values of the same length as language_list_a, where 1. means "is present in
    language_list_b", and 0. means "not present".
    """
    if len(language_list_a) != len(language_list_b):
        raise ValueError("The two language lists should be of the same size")
    new_log_prior = np.zeros(len(priors))
    checks_per_lang = np.zeros(len(language_list_a))
    for i in range(len(language_list_a)):
        for j in range(len(language_list_b)):
            if language_list_a[i] == language_list_b[j]:
                checks_per_lang[i] = 1.
                new_log_prior[i] = priors[j]
    return checks_per_lang, new_log_prior


checks_per_language, new_log_prior = check_all_lang_lists_against_each_other(all_langs_as_in_simlang, languages)
print('')
print('')
print("checks_per_language is:")
print(checks_per_language)
print("np.sum(checks_per_language) is:")
print(np.sum(checks_per_language))


print('')
print('')
print("new_log_prior is:")
print(new_log_prior)
print("np.exp(new_log_prior) is:")
print(np.exp(new_log_prior))
print("new_log_prior.shape is:")
print(new_log_prior.shape)
print("np.exp(logsumexp(new_log_prior)) is:")
print(np.exp(logsumexp(new_log_prior)))
priors = new_log_prior

# Ok, this shows that for each language in the list of all_possible_languages generated by my own code, there is a
# corresponding languages in the code from SimLang lab 21, so instead there must be something wrong with the way I
# categorise the languages. Firstly, it looks like my classify_language() function underestimates the number of
# compositional languages. So let's first have a look at which languages it classifies as compositional:


compositional_langs_indices_my_code = np.where(class_per_lang==3)[0]
print('')
print('')
print("compositional_langs_indices_my_code MY CODE are:")
print(compositional_langs_indices_my_code)
print("len(compositional_langs_indices_my_code) MY CODE are:")
print(len(compositional_langs_indices_my_code))


for index in compositional_langs_indices_my_code:
    print('')
    print("index MY CODE is:")
    print(index)
    print("all_possible_languages[index] MY CODE is:")
    print(all_possible_languages[index])


# And now let's do the same for the languages from SimLang Lab 21:

compositional_langs_indices_simlang = np.where(np.array(types)==3)[0]
print('')
print('')
print("compositional_langs_indices_simlang SIMLANG CODE are:")
print(compositional_langs_indices_simlang)
print("len(compositional_langs_indices_simlang) SIMLANG CODE are:")
print(len(compositional_langs_indices_simlang))


for index in compositional_langs_indices_simlang:
    print('')
    print("index SIMLANG CODE is:")
    print(index)
    print("languages[index] SIMLANG CODE is:")
    print(languages[index])


# Hmm, so it looks like instead of there being a bug in my code, there might actually be a bug in the SimLang lab 21
# code (or rather, in the code that generated the list of types that was copied into SimLang lab 21)








# A reproduction of the production function of Kirby et al. (2015):


# first we need to write a quick function that removes every instance of a given element from a list (to use for
# removing the 'correct' forms_without_noisy_variants from a list of possible forms_without_noisy_variants for a given topic:
def remove_all_instances(my_list, element_to_be_removed):
    """
    Takes a list, and removes all instances of a given element from it

    :param my_list: a list
    :param element_to_be_removed: the element to be removed; can be of any type
    :return: the list with all instances of the target element removed
    """
    i = 0  # loop counter
    length = len(my_list)  # list length
    while (i < len(my_list)):
        if (my_list[i] == element_to_be_removed):
            my_list.remove(my_list[i])
            # as an element is removed
            # so decrease the length by 1
            length = length - 1
            # run loop again to check element
            # at same index, when item removed
            # next item will shift to the left
            continue
        i = i + 1
    return my_list


# Now let's define a function that calculates the probabilities of producing each of the possible forms_without_noisy_variants, given a particular language and topic:
def production_probs_kirby_et_al(language, topic, gamma, error):
    """
    Calculates the production probabilities for each of the possible forms_without_noisy_variants given a language and topic, as defined by Kirby et al. (2015)

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable), where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the speaker intends to communicate
    :param gamma: parameter that determines the strength of the penalty on ambiguity
    :param error: the probability of making an error in production
    :return: 1D numpy array containing a production probability for each possible form (where the index of the probability corresponds to the index of the form in the global variable "forms_without_noisy_variants")
    """
    for m in range(len(meanings)):
        if meanings[m] == topic:
            topic_index = m
    correct_form = language[topic_index]
    error_forms = list(forms_without_noise)  #This may seem a bit weird, but a speaker should be able to produce *any* form as an error form right? Not limited to only the other forms that exist within their language? (Otherwise a speaker with a degenerate language could never make a production error).
    error_forms = remove_all_instances(error_forms, correct_form)
    if len(error_forms) == 0:  # if the list of error_forms is empty because the language is degenerate
        error_forms = language  # simply choose an error_form from the whole language
    ambiguity = 0
    for f in language:
        if f == correct_form:
            ambiguity += 1
    prop_to_prob_correct_form = ((1./ambiguity)**gamma)*(1.-error)
    prop_to_prob_error_form = error / (len(forms_without_noise) - 1)
    prop_to_prob_per_form_array = np.zeros(len(forms_without_noise))
    for i in range(len(forms_without_noise)):
        if forms_without_noise[i] == correct_form:
            prop_to_prob_per_form_array[i] = prop_to_prob_correct_form
        else:
            prop_to_prob_per_form_array[i] = prop_to_prob_error_form
    return prop_to_prob_per_form_array




def create_noisy_variants(form):
    """
    Takes a form and generates all its possible noisy variants. NOTE however that in its current form, this function only creates noisy variants in which only a single element of the original form is replaced with a blank! (So it creates for instance 'a_' and '_b', but not '__'.

    :param form: a form (string)
    :return: a list of possible noisy variants of that form
    """
    noisy_variant_list = []
    for i in range(len(form)):
        noisy_variant = form[:i] + '_' + form[i+1:]
        # Instead of string slicing, another way of doing this would be to convert the string into a list, replace the element at the ith index, and then convert it back into a string using the 'join' method, see: https://www.quora.com/How-do-you-change-one-character-in-a-string-in-Python
        noisy_variant_list.append(noisy_variant)
    return noisy_variant_list



def production_probs_with_noise(language, topic, gamma, error, noise_prob):
    """
    Calculates the production probabilities for each of the possible forms (including both forms without noise and all possible noisy variants) given a language and topic, and the probability of environmental noise

    :param language: list of forms that has same length as list of meanings (global variable), where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the speaker intends to communicate
    :param gamma: parameter that determines the strength of the penalty on ambiguity
    :param error: the probability of making an error in production
    :param noise_prob: the probability of environmental noise masking part of the utterance
    :return: 1D numpy array containing a production probability for each possible form (where the index of the probability corresponds to the index of the form in the global variable "all_forms_including_noisy_variants")
    """

    # print('')
    # print("all_forms_including_noisy_variants are:")
    # print(all_forms_including_noisy_variants)
    # print("len(all_forms_including_noisy_variants) is:")
    # print(len(all_forms_including_noisy_variants))

    for m in range(len(meanings)):
        if meanings[m] == topic:
            topic_index = m
    correct_form = language[topic_index]
    error_forms = list(forms_without_noise)  #This may seem a bit weird, but a speaker should be able to produce *any* form as an error form right? Not limited to only the other forms that exist within their language? (Otherwise a speaker with a degenerate language could never make a production error).
    error_forms = remove_all_instances(error_forms, correct_form)
    if len(error_forms) == 0:  # if the list of error_forms is empty because the language is degenerate
        error_forms = language  # simply choose an error_form from the whole language

    # print("language is:")
    # print(language)
    #
    # print('')
    # print("topic is:")
    # print(topic)
    #
    # print('')
    # print("correct_form is:")
    # print(correct_form)
    #
    # print('')
    # print("error_forms are:")
    # print(error_forms)

    noisy_variants_correct_form = create_noisy_variants(correct_form)
    # print("noisy_variants_correct_form are:")
    # print(noisy_variants_correct_form)

    noisy_variants_error_forms = []
    for error_form in error_forms:
        noisy_variants = create_noisy_variants(error_form)
        noisy_variants_error_forms = noisy_variants_error_forms+noisy_variants
    # print("noisy_variants_error_forms are:")
    # print(noisy_variants_error_forms)

    ambiguity = 0
    for f in language:
        if f == correct_form:
            ambiguity += 1

    prop_to_prob_correct_form_complete = ((1./ambiguity)**gamma)*(1.-error)*(1 - noise_prob)

    prop_to_prob_error_form_complete = error / (len(forms_without_noise) - 1)*(1 - noise_prob)

    prop_to_prob_correct_form_noisy = ((1. / ambiguity) ** gamma) * (1. - error) * (noise_prob / len(noisy_forms))

    prop_to_prob_error_form_noisy = error / (len(forms_without_noise) - 1) * (1 - noise_prob) * (noise_prob / len(noisy_forms))

    # print('')
    # print('')
    prop_to_prob_per_form_array = np.zeros(len(all_forms_including_noisy_variants))
    for i in range(len(all_forms_including_noisy_variants)):
        # print('')
        # print("all_forms_including_noisy_variants[i] is:")
        # print(all_forms_including_noisy_variants[i])
        if all_forms_including_noisy_variants[i] == correct_form:
            # print('all_forms_including_noisy_variants[i] == correct_form !')
            prop_to_prob_per_form_array[i] = prop_to_prob_correct_form_complete
        elif all_forms_including_noisy_variants[i] in noisy_variants_correct_form:
            # print('all_possible_forms[i] in noisy_variants_correct_form !')
            prop_to_prob_per_form_array[i] = prop_to_prob_correct_form_noisy
        elif all_forms_including_noisy_variants[i] in noisy_variants_error_forms:
            # print('all_possible_forms[i] in noisy_variants_error_forms !')
            prop_to_prob_per_form_array[i] = prop_to_prob_error_form_noisy
        else:
            # print('all_possible_forms[i] must be a complete error form !')
            prop_to_prob_per_form_array[i] = prop_to_prob_error_form_complete
        # print("prob_per_form_array[i] is:")
        # print(prob_per_form_array[i])
    return prop_to_prob_per_form_array



# print('')
# print('')
# print('THIS IS THE PRODUCTION_PROBS_WITH_NOISE() AT WORK:')
# production_probs_array_with_noise = production_probs_with_noise(other_lang, "02", gamma, error, noise_prob)
# print("production_probs_array_with_noise are:")
# print(production_probs_array_with_noise)
# print("np.sum(production_probs_array_with_noise) are:")
# print(np.sum(production_probs_array_with_noise))
# print("len(production_probs_array_with_noise) are:")
# print(len(production_probs_array_with_noise))
#


# And finally, let's write a function that actually produces an utterance, given a language and a topic
def produce(language, topic, gamma, error):
    """
    Produces an actual utterance, given a language and a topic

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable), where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the speaker intends to communicate
    :param gamma: parameter that determines the strength of the penalty on ambiguity
    :param error: the probability of making an error in production
    :return: an utterance. That is, a single form chosen from either the global variable "forms_without_noise" (if noise is False) or the global variable "all_forms_including_noisy_variants" (if noise is True).
        """
    if noise:
        prop_to_prob_per_form_array = production_probs_with_noise(language, topic, gamma, error, noise_prob)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(all_forms_including_noisy_variants, p=prob_per_form_array)
    else:
        prop_to_prob_per_form_array = production_probs_kirby_et_al(language, topic, gamma, error)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(forms_without_noise, p=prob_per_form_array)
    return utterance




def receive(language, utterance):
    """
    Takes a language and an utterance, and returns an interpretation of that utterance, following the language

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable), where each form is mapped to the meaning at the corresponding index
    :param utterance: a form (string)
    :return: an interpretation (string)
    """
    possible_interpretations = []
    for i in range(len(language)):
        if language[i] == utterance:
            possible_interpretations.append(meanings[i])
    interpretation = random.choice(possible_interpretations)
    return interpretation




def update_posterior(log_posterior, topic, utterance):
    # First, let's find out what the index of the utterance is in the list of all possible forms (including the noisy variants):
    for i in range(len(all_forms_including_noisy_variants)):
        if all_forms_including_noisy_variants[i] == utterance:
            utterance_index = i
    # Now, let's go through each hypothesis (i.e. language), and update its posterior probability given the <topic, utterance> pair that was given as input:
    new_log_posterior = []
    for j in range(len(log_posterior)):
        hypothesis = all_possible_languages[j]
        if noise:
            likelihood_per_form_array = production_probs_with_noise(hypothesis, topic, gamma, error, noise_prob)
        else:
            likelihood_per_form_array = production_probs_kirby_et_al(hypothesis, topic, gamma, error)
        log_likelihood_per_form_array = np.log(likelihood_per_form_array)
        for m in range(len(meanings)):
            if meanings[m] == topic:
                topic_index = m
        correct_form = hypothesis[topic_index]
        noisy_variants_correct_form = create_noisy_variants(correct_form)
        new_log_posterior.append(log_posterior[j] + log_likelihood_per_form_array[utterance_index])

    new_log_posterior_normalized = np.subtract(new_log_posterior, logsumexp(new_log_posterior))

    return new_log_posterior_normalized


# print('')
# print('')
# example_prior = np.array([1./len(all_possible_languages) for x in range(len(all_possible_languages))])
# example_log_prior = np.log(example_prior)
# print("np.exp(example_log_prior) is:")
# print(np.exp(example_log_prior))
# print("np.exp(logsumexp(example_log_prior)) is:")
# print(np.exp(logsumexp(example_log_prior)))
#
# print('')
# print('')
# print("NOW, LET'S TEST THE UPDATE_POSTERIOR() FUNCTION:")
# topic = '02'
# utterance = 'a_'
# new_log_posterior_normalized = update_posterior(example_log_prior, topic, utterance)
# print('')
# print('')
# print("np.exp(new_log_posterior_normalized) is:")
# print(np.exp(new_log_posterior_normalized))
# print("new_log_posterior_normalized.shape is:")
# print(new_log_posterior_normalized.shape)
# print("np.exp(logsumexp(new_log_posterior_normalized)) is:")
# print(np.exp(logsumexp(new_log_posterior_normalized)))



def new_population(popsize):
    """
    Creates a new population of agents, where each agent simply consists of the prior probability distribution (which is assumed to be defined as a global variable called 'priors')

    :param popsize: the number of agents desired in the new population
    :return: 2D numpy array, with agents on the rows, and hypotheses (or rather their corresponding prior probabilities) on the columns.
    """
    population = [priors for x in range(popsize)]
    population = np.array(population)
    return population





def log_roulette_wheel(normedlogs):
    r=np.log(random.random()) #generate a random number in [0,1), then convert to log
    accumulator = normedlogs[0]
    for i in range(len(normedlogs)):
        if r < accumulator:
            return i
        accumulator = logsumexp([accumulator, normedlogs[i + 1]])


def sample(posterior):
    return all_possible_languages[log_roulette_wheel(posterior)]


def population_communication(population, rounds):
    data = []
    for i in range(rounds):
        speaker_index = random.randrange(len(population))
        hearer_index = random.randrange(len(population) - 1)
        if hearer_index >= speaker_index:
            hearer_index += 1
        meaning = random.choice(meanings)
        signal = produce(sample(population[speaker_index]), meaning, gamma, error)
        population[hearer_index] = update_posterior(population[hearer_index], meaning, signal)
        data.append((meaning, signal))
    return data


def language_stats(posteriors):
    stats = [0., 0., 0., 0.]  # degenerate, holistic, other, compositional
    for p in posteriors:
        for i in range(len(p)):
            stats[int(class_per_lang[i])] += np.exp(p[i]) / len(posteriors)
    return stats


def simulation(generations, rounds, bottleneck, popsize, data):
    results = []
    population = new_population(popsize)

    for i in range(generations):
        print('.')
        for j in range(popsize):
            for k in range(bottleneck):
                meaning, signal = random.choice(data)
                population[j] = update_posterior(population[j], meaning, signal)
        data = population_communication(population, rounds)
        results.append(language_stats(population))
        if turnover:
            population = new_population(popsize)

    return results, data



def plot_graph(results, fig_title):

    average_degenerate = []
    average_holistic = []
    average_compositional = []

    for i in range(len(results[0])):
        total_degenerate = 0
        total_holistic = 0
        total_compositional = 0
        for result in results:
            total_degenerate += result[i][0]
            total_holistic += result[i][1]
            total_compositional += result[i][3]
        average_degenerate.append(total_degenerate / len(results))
        average_holistic.append(total_holistic / len(results))
        average_compositional.append(total_compositional / len(results))

    plt.plot(average_degenerate, color='orange', label='degenerate')
    plt.plot(average_holistic, color='green', label='holistic')
    plt.plot(average_compositional, color='purple', label='compositional')
    plt.xlabel('generations')
    plt.ylabel('proportion')
    plt.legend()
    plt.grid()
    plt.savefig(fig_title+".pdf")
    plt.show()


initial = [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')]
gamma = 2  # parameter that determines strength of ambiguity penalty (Kirby et al. used gamma = 2 for communication
            # condition and gamma = 0 for condition without communication)
turnover = False  # determines whether new individuals enter the population or not
results = []
for i in range(10):
    results.append(simulation(200, 20, 20, 2, initial)[0])
fig_title = "Plot_avoid_ambiguity_gamma_"+str(gamma)+"_turnover_"+str(turnover)
plot_graph(results, fig_title)


