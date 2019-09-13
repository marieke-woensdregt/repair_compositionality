import itertools
import numpy as np
import random
import scipy.misc
import time
from copy import deepcopy
from math import log
import pickle
import sys


# FROM SIMLANG LAB 21:
languages_simlang = [[('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')]]
types_simlang = [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]
priors_simlang = [-0.9178860550328204, -10.749415928290118, -10.749415928290118, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -12.460704095246543, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -12.460704095246543, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -2.304180416152711, -11.272664072079987, -10.749415928290118, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -10.749415928290118, -11.272664072079987, -2.304180416152711, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -12.460704095246543, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -12.460704095246543, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -10.749415928290118, -10.749415928290118, -0.9178860550328204]


# MY OWN CODE:

# First some parameters:
meanings = ['02', '03', '12', '13']  # all possible meanings
forms_without_noise = ['aa', 'ab', 'ba', 'bb']  # all possible forms, excluding their possible 'noisy variants'
noisy_forms = ['a_', 'b_', '_a', '_b']  # all possible noisy variants of the forms above
all_forms_including_noisy_variants = forms_without_noise+noisy_forms  # all possible forms, including both complete
# forms and noisy variants
error = 0.05  # the probability of making a production error (Kirby et al., 2015 use 0.05)



# Some functions to create and classify all possible languages:
def create_all_possible_languages(meanings, forms):
    """Creates all possible languages

    :param meanings: list of strings corresponding to all possible meanings
    :type meanings: list
    :param forms: list of strings corresponding to all possible forms_without_noisy_variants
    :type forms: list
    :returns: list of tuples which represent languages, where each tuple consists of forms_without_noisy_variants and
    has length len(meanings)
    :rtype: list
    """
    all_possible_languages = list(
        itertools.product(forms, repeat=len(meanings)))
    return all_possible_languages


def classify_language(lang, forms, meanings):
    """
    Classify one particular language as either 'degenerate' (0), 'holistic' (1), 'other' (2)
    or 'compositional' (3) (Kirby et al., 2015)

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :type lang: tuple
    :param forms: list of strings corresponding to all possible forms_without_noisy_variants
    :type forms: list
    :returns: integer corresponding to category that language belongs to:
    0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, 4 = other (here I'm following the
    ordering used in the Kirby et al., 2015 paper; NOT the ordering from SimLang lab 21)
    :rtype: int
    """
    # TODO: See if I can modify this function so that it can deal with any number of forms_without_noisy_variants and
    #  meanings.
    class_degenerate = 0
    class_holistic = 1
    class_hybrid = 2  # this is a hybrid between a holistic and a compositional language; where *half* of the partial
    # forms is mapped consistently to partial meanings (instead of that being the case for *all* partial forms)
    class_compositional = 3
    class_other = 4

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

    # lang is compositional if it makes use of all possible forms_without_noisy_variants, *and* each form element maps
    # to the same meaning element for each form:
    elif forms[0] in lang and forms[1] in lang and forms[2] in lang and forms[
        3] in lang and lang[0][0] == lang[1][0] and lang[2][0] == lang[3][0] and lang[0][
        1] == lang[2][1] and lang[1][1] == lang[3][1]:
        return class_compositional

    # lang is holistic if it is *not* compositional, but *does* make use of all possible forms_without_noisy_variants:
    elif forms[0] in lang and forms[1] in lang and forms[2] in lang and forms[3] in lang:
        # within holistic languages, we can distinguish between those in which at least one part form is mapped
        # consistently onto one part meaning. This class we will call 'hybrid' (because for the purposes of repair, it
        # is a hybrid between a holistic and a compositional language, because for half of the possible noisy forms that
        # a listener could receive it allows the listener to figure out *part* of the meaning, and therefore use a
        # restricted request for repair instead of an open request.
        if lang[0][0] == lang[1][0] and lang[2][0] == lang[3][0]:
            return class_hybrid
        elif lang[0][1] == lang[2][1] and lang[1][1] == lang[3][1]:
            return class_hybrid
        else:
            return class_holistic

    # In all other cases, a language belongs to the 'other' category:
    else:
        return class_other



all_possible_languages = create_all_possible_languages(meanings, forms_without_noise)



def classify_all_languages(language_list):
    """
    Classify all languages as either 'degenerate' (0), 'holistic' (1), 'other' (2) or 'compositional' (3)
    (Kirby et al., 2015)

    :param language_list: list of all languages
    :type language_list: list
    :returns: 1D numpy array containing integer corresponding to category of corresponding
    language index as hardcoded in classify_language function: 0 = degenerate, 1 = holistic, 2 = hybrid,
    3 = compositional, 4 = other (here I'm following the ordering used in the Kirby et al., 2015 paper; NOT the ordering
    from SimLang lab 21)
    :rtype: 1D numpy array
    """
    class_per_lang = np.zeros(len(language_list))
    for l in range(len(language_list)):
        class_per_lang[l] = classify_language(language_list[l], forms_without_noise, meanings)
    return class_per_lang




class_per_lang = classify_all_languages(all_possible_languages)

no_of_each_class = np.bincount(class_per_lang.astype(int))




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
    for l in range(len(language_list)):
        lang_as_in_simlang = [(meanings[x], language_list[l][x]) for x in range(len(meanings))]
        all_langs_as_in_simlang.append(lang_as_in_simlang)
    return all_langs_as_in_simlang




all_langs_as_in_simlang = transform_all_languages_to_simlang_format(all_possible_languages)




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
    new_log_prior = np.zeros(len(priors_simlang))
    checks_per_lang = np.zeros(len(language_list_a))
    for i in range(len(language_list_a)):
        for j in range(len(language_list_b)):
            if language_list_a[i] == language_list_b[j]:
                checks_per_lang[i] = 1.
                new_log_prior[i] = priors_simlang[j]
    return checks_per_lang, new_log_prior




checks_per_language, new_log_prior = check_all_lang_lists_against_each_other(all_langs_as_in_simlang, languages_simlang)




# A reproduction of the production function of Kirby et al. (2015):

# Now let's define a function that calculates the probabilities of producing each of the possible forms_without_noisy_
# variants, given a particular language and topic:
def production_likelihoods_kirby_et_al(language, topic, gamma, error):
    """
    Calculates the production probabilities for each of the possible forms_without_noisy_variants given a language and
    topic, as defined by Kirby et al. (2015)

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param gamma: parameter that determines the strength of the penalty on ambiguity
    :param error: the probability of making an error in production
    :return: 1D numpy array containing a production probability for each possible form (where the index of the
    probability corresponds to the index of the form in the global variable "forms_without_noisy_variants")
    """
    for m in range(len(meanings)):
        if meanings[m] == topic:
            topic_index = m
    correct_form = language[topic_index]
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
    Takes a form and generates all its possible noisy variants. NOTE however that in its current form, this function
    only creates noisy variants in which only a single element of the original form is replaced with a blank! (So it
    creates for instance 'a_' and '_b', but not '__'.)

    :param form: a form (string)
    :return: a list of possible noisy variants of that form
    """
    noisy_variant_list = []
    for i in range(len(form)):
        noisy_variant = form[:i] + '_' + form[i+1:]
        # Instead of string slicing, another way of doing this would be to convert the string into a list, replace the
        # element at the ith index, and then convert it back into a string using the 'join' method,
        # see: https://www.quora.com/How-do-you-change-one-character-in-a-string-in-Python
        noisy_variant_list.append(noisy_variant)
    return noisy_variant_list


# we also need a function that removes every instance of a given element from a list (to use for
# removing the 'correct' forms_without_noisy_variants from a list of possible forms_without_noisy_variants for a given
# topic:
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


def production_likelihoods_with_noise(language, topic, gamma, error, noise_prob):
    """
    Calculates the production probabilities for each of the possible forms (including both forms without noise and all
    possible noisy variants) given a language and topic, and the probability of environmental noise

    :param language: list of forms that has same length as list of meanings (global variable), where each form is
    mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param gamma: parameter that determines the strength of the penalty on ambiguity
    :param error: the probability of making an error in production
    :param noise_prob: the probability of environmental noise masking part of the utterance
    :return: 1D numpy array containing a production probability for each possible form (where the index of the
    probability corresponds to the index of the form in the global variable "all_forms_including_noisy_variants")
    """
    for m in range(len(meanings)):
        if meanings[m] == topic:
            topic_index = m
    correct_form = language[topic_index]
    error_forms = list(forms_without_noise)  # This may seem a bit weird, but a speaker should be able to produce *any*
    # form as an error form right? Not limited to only the other forms that exist within their language? (Otherwise a
    # speaker with a degenerate language could never make a production error).
    error_forms = remove_all_instances(error_forms, correct_form)
    if len(error_forms) == 0:  # if the list of error_forms is empty because the language is degenerate
        error_forms = language  # simply choose an error_form from the whole language
    noisy_variants_correct_form = create_noisy_variants(correct_form)
    noisy_variants_error_forms = []
    for error_form in error_forms:
        noisy_variants = create_noisy_variants(error_form)
        noisy_variants_error_forms = noisy_variants_error_forms+noisy_variants
    ambiguity = 0
    for f in language:
        if f == correct_form:
            ambiguity += 1
    prop_to_prob_correct_form_complete = ((1./ambiguity)**gamma)*(1.-error)*(1 - noise_prob)
    prop_to_prob_error_form_complete = error / (len(forms_without_noise) - 1)*(1 - noise_prob)
    prop_to_prob_correct_form_noisy = ((1. / ambiguity) ** gamma) * (1. - error) * (noise_prob / len(noisy_forms))
    prop_to_prob_error_form_noisy = error/(len(forms_without_noise)-1) * (1-noise_prob)*(noise_prob/len(noisy_forms))
    prop_to_prob_per_form_array = np.zeros(len(all_forms_including_noisy_variants))
    for i in range(len(all_forms_including_noisy_variants)):
        if all_forms_including_noisy_variants[i] == correct_form:
            prop_to_prob_per_form_array[i] = prop_to_prob_correct_form_complete
        elif all_forms_including_noisy_variants[i] in noisy_variants_correct_form:
            prop_to_prob_per_form_array[i] = prop_to_prob_correct_form_noisy
        elif all_forms_including_noisy_variants[i] in noisy_variants_error_forms:
            prop_to_prob_per_form_array[i] = prop_to_prob_error_form_noisy
        else:
            prop_to_prob_per_form_array[i] = prop_to_prob_error_form_complete
    return prop_to_prob_per_form_array



# And finally, let's write a function that actually produces an utterance, given a language and a topic
def produce(language, topic, gamma, error, noise_switch):
    """
    Produces an actual utterance, given a language and a topic

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param gamma: parameter that determines the strength of the penalty on ambiguity
    :param error: the probability of making an error in production
    :param noise_switch: turns noise on when set to True, and off when set to False
    :return: an utterance. That is, a single form chosen from either the global variable "forms_without_noise" (if
    noise is False) or the global variable "all_forms_including_noisy_variants" (if noise is True).
        """
    if noise_switch:
        prop_to_prob_per_form_array = production_likelihoods_with_noise(language, topic, gamma, error, noise_prob)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(all_forms_including_noisy_variants, p=prob_per_form_array)
    else:
        prop_to_prob_per_form_array = production_likelihoods_kirby_et_al(language, topic, gamma, error)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(forms_without_noise, p=prob_per_form_array)
    return utterance




def produce_simlang(language, meaning):
    """
    This function is copied directly from lab 21 of the SimLang course of 2019. I only renamed and reformatted some of
    the variables below, in order to make it work with my code (indicated with comments).

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param meaning: a meaning (string) that the speaker wants to communicate
    :return: an utterance (string)
    """

    # Added by me:
    if gamma > 0.:
        communication = True
    else:
        communication = False
    signals = forms_without_noise
    noise = error
    language_simlang_style = []
    for i in range(len(language)):
        language_simlang_style.append((meanings[i], language[i]))

    for m, s in language_simlang_style:
        if m == meaning:
            signal = s
    if communication:
        speaker_meaning = receive_without_repair(language, signal)  # I changed this to receive_without_repair() instead
                                                                    # of receive()
        if speaker_meaning != meaning:
            signal = random.choice(signals)
    if random.random() < noise:
        other_signals = deepcopy(signals)
        other_signals.remove(signal)
        return random.choice(other_signals)
    return signal




def receive_without_repair(language, utterance):
    """
    Takes a language and an utterance, and returns an interpretation of that utterance, following the language

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param utterance: a form (string)
    :return: an interpretation (string)
    """
    possible_interpretations = []
    for i in range(len(language)):
        if language[i] == utterance:
            possible_interpretations.append(meanings[i])
    if len(possible_interpretations) == 0:
        possible_interpretations = meanings
    interpretation = random.choice(possible_interpretations)
    return interpretation





def noisy_to_complete_forms(noisy_form, forms_without_noise):
    """
    Takes a noisy form and returns all possible complete forms that it's compatible with.

    :param noisy_form: a noisy form (i.e. a string containing '_' as at least one of the characters)
    :param forms_without_noise: The full set of possible complete forms
    :return: A list of complete forms that the noisy form is compatible with
    """
    possible_complete_forms = []
    amount_of_noise = noisy_form.count('_')
    for complete_form in forms_without_noise:
        similarity_score = 0
        for i in range(len(noisy_form)):
            if noisy_form[i] == complete_form[i]:
                similarity_score += 1
        if similarity_score == len(complete_form)-amount_of_noise:
            possible_complete_forms.append(complete_form)
    return possible_complete_forms



def find_possible_interpretations(language, forms):
    """
    Finds all meanings that the forms given as input are mapped to in the language given as input

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param forms: list of forms
    :return: list of meanings (type: string) that the forms given as input are mapped to in the language given as input
    """
    possible_interpretations = []
    for i in range(len(language)):
        if language[i] in forms:
            possible_interpretations.append(meanings[i])
    return possible_interpretations



def find_partial_meaning(language, noisy_form):
    """
    Checks whether the noisy_form given as input maps unambiguously to a partial meaning in the language given as
    input, and if so, returns that partial meaning.

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param noisy_form: a noisy form (i.e. a string containing '_' as at least one of the characters)
    :return: a list containing the partial meaning that the noisy_form maps unambiguously to, if there is one
    """
    part_meanings_as_ints = []
    for i in range(len(meanings)):
        for j in range(len(meanings[0])):
            part_meanings_as_ints.append(int(meanings[i][j]))
    max_part_meaning = max(part_meanings_as_ints)
    count_per_partial_meaning = np.zeros(max_part_meaning+1)
    for i in range(len(noisy_form)):
        if noisy_form[i] != '_':
            for j in range(len(language)):
                if language[j][i] == noisy_form[i]:
                    count_per_partial_meaning[int(meanings[j][i])] += 1
    n_features = 0
    for i in range(len(meanings)):
        if meanings[i][0] == meanings[0][0]:
            n_features += 1
    if np.sum(count_per_partial_meaning) == n_features:
        part_meaning_index = np.where(count_per_partial_meaning==n_features)[0]
    else:
        part_meaning_index = []
    if len(part_meaning_index) == 1:
        return part_meaning_index
    else:
        return []



#TODO: This has turned into a bit of a monster function. Maybe shorten it by pulling out the code that calculates the
# probabilities for the different response options and putting that in a seperate function?
def receive_with_repair(language, utterance):
    """
    Receives and utterance and gives a response, which can either be an interpretation or a repair initiator. How likely
    these two response types are to happen depends on the settings of the paremeters 'mutual_understanding' and
    'minimal_effort' (and, if minimal_effort is set to True, the parameter 'cost_vector'). These three parameters are
    all assumed to be global variables.

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param utterance: an utterance (string)
    :return: a response, which can either be an interpretation (i.e. meaning) or a repair initiator. A repair initiator
    can be of two types: if the listener has grasped part of the meaning, it will be a restricted request, which is a
    string containing the partial meaning that the listener did grasp, followed by a question mark. If the listener did
    not grasp any of the meaning, it will be an open request, which is simply '??'
    """
    if not mutual_understanding and not minimal_effort:
        raise ValueError(
            "Sorry, this function has only been implemented for at least one of either mutual_understanding or minimal_effort being True"
        )
    if '_' in utterance:
        compatible_forms = noisy_to_complete_forms(utterance, forms_without_noise)
        possible_interpretations = find_possible_interpretations(language, compatible_forms)
        if len(possible_interpretations) == 0:
            possible_interpretations = meanings
        partial_meaning = find_partial_meaning(language, utterance)
        if mutual_understanding and minimal_effort:
            prop_to_prob_no_repair = (1./len(possible_interpretations))-cost_vector[0]
            if len(partial_meaning) == 1:
                prop_to_prob_repair = (1.-(1./len(possible_interpretations)))-cost_vector[1]
                repair_initiator = str(partial_meaning[0])+'?'
            elif len(partial_meaning) == 0:
                prop_to_prob_repair = (1.-(1./len(possible_interpretations)))-cost_vector[2]
                repair_initiator = '??'
        elif mutual_understanding and not minimal_effort:
            if len(possible_interpretations) > 1:
                prop_to_prob_no_repair = 0.
                prop_to_prob_repair = 1.
                if len(partial_meaning) == 1:
                    repair_initiator = str(partial_meaning[0])+'?'
                elif len(partial_meaning) == 0:
                    repair_initiator = '??'
            elif len(possible_interpretations) == 1:
                prop_to_prob_no_repair = 1.
                prop_to_prob_repair = 0.
        elif not mutual_understanding and minimal_effort:
            prop_to_prob_no_repair = 1.
            prop_to_prob_repair = 0.
            if len(partial_meaning) == 1:
                repair_initiator = str(partial_meaning[0])+'?'
            elif len(partial_meaning) == 0:
                repair_initiator = '??'
        prop_to_prob_per_response = np.array([prop_to_prob_no_repair, prop_to_prob_repair])
        for i in range(len(prop_to_prob_per_response)):
            if prop_to_prob_per_response[i] < 0.0:
                prop_to_prob_per_response[i] = 0.0
        normalized_response_probs = np.divide(prop_to_prob_per_response, np.sum(prop_to_prob_per_response))
        selected_response = np.random.choice(np.arange(2), p=normalized_response_probs)
        if selected_response == 0:
            response = random.choice(possible_interpretations)
        elif selected_response == 1:
            response = repair_initiator
    else:
        response = receive_without_repair(language, utterance)
    return response




def update_posterior(log_posterior, topic, utterance):
    """
    Takes a LOG posterior probability distribution and a <topic, utterance> pair, and updates the posterior probability
    distribution accordingly

    :param log_posterior: 1D numpy array containing LOG posterior probability values for each hypothesis
    :param topic: a topic (string from the global variable meanings)
    :param utterance: an utterance (string from the global variable forms (can be a noisy form if parameter noise is
    True)
    :return: the updated (and normalized) log_posterior (1D numpy array)
    """
    # First, let's find out what the index of the utterance is in the list of all possible forms (including the noisy
    # variants):
    for i in range(len(all_forms_including_noisy_variants)):
        if all_forms_including_noisy_variants[i] == utterance:
            utterance_index = i
    # Now, let's go through each hypothesis (i.e. language), and update its posterior probability given the
    # <topic, utterance> pair that was given as input:
    new_log_posterior = []
    for j in range(len(log_posterior)):
        hypothesis = all_possible_languages[j]
        if noise:
            likelihood_per_form_array = production_likelihoods_with_noise(hypothesis, topic, gamma, error, noise_prob)
        else:
            likelihood_per_form_array = production_likelihoods_kirby_et_al(hypothesis, topic, gamma, error)
        log_likelihood_per_form_array = np.log(likelihood_per_form_array)
        new_log_posterior.append(log_posterior[j] + log_likelihood_per_form_array[utterance_index])

    new_log_posterior_normalized = np.subtract(new_log_posterior, scipy.misc.logsumexp(new_log_posterior))

    return new_log_posterior_normalized




def normalize_logprobs_simlang(logprobs):
    """
    This function is copied directly from lab 21 of the SimLang course of 2019.

    :param logprobs: a list of LOG probabilities
    :return: a list of normalised LOG probabilities
    """
    logtotal = scipy.misc.logsumexp(logprobs) #calculates the summed log probabilities
    normedlogs = []
    for logp in logprobs:
        normedlogs.append(logp - logtotal) #normalise - subtracting in the log domain equivalent to divising in the
                                            # normal domain
    return normedlogs



def update_posterior_simlang(posterior, meaning, signal):
    """
    This function is copied directly from lab 21 of the SimLang course of 2019. I only renamed some of the variables
    below, in order to make it work with my code (under the "# Added by me" comment).

    :param posterior: a list of LOG posterior probabilities
    :param meaning: the meaning from the meaning-signal pair that was observed (string)
    :param signal: the signal from the meaning-signal pair that was observed (string)
    :return: a list of normalised LOG posterior probabilities, updated based on the meaning-signal pair that was
    observed
    """

    # added by me:
    signals = forms_without_noise
    noise = error

    in_language = log(1 - noise)
    out_of_language = log(noise / (len(signals) - 1))
    new_posterior = []
    for i in range(len(posterior)):
        if (meaning, signal) in all_langs_as_in_simlang[i]:
            new_posterior.append(posterior[i] + in_language)
        else:
            new_posterior.append(posterior[i] + out_of_language)
    return normalize_logprobs_simlang(new_posterior)



def new_population(popsize):
    """
    Creates a new population of agents, where each agent simply consists of the prior probability distribution (which
    is assumed to be defined as a global variable called 'priors')

    :param popsize: the number of agents desired in the new population
    :return: 2D numpy array, with agents on the rows, and hypotheses (or rather their corresponding LOG prior
    probabilities)
    on the columns.
    """
    population = [priors for x in range(popsize)]
    population = np.array(population)
    return population



def log_roulette_wheel(normedlogs):
    """
    Samples an index from a list of LOG probabilities, where each index has a probability proportional to their
    probability of being chosen

    :param normedlogs: a list of normalized LOG probabilities
    :return: an index somewhere between 0 and len(normedlogs)
    """
    r = np.log(random.random())  # generate a random number in [0,1), then convert to log
    accumulator = normedlogs[0]
    for i in range(len(normedlogs)):
        if r < accumulator:
            return i
        accumulator = scipy.misc.logsumexp([accumulator, normedlogs[i + 1]])


def sample(log_posterior):
    """
    Samples a language based on the posterior

    :param log_posterior: a list of LOG posterior probabilities
    :return: a language (list of forms_without_noisy_variants that has same length as the global variable meanings,
    where each form is mapped to the meaning at the corresponding index)
    """
    return all_possible_languages[log_roulette_wheel(log_posterior)]


def population_communication(population, rounds):
    """
    Takes a population, makes it communicate for a number of rounds (where agents' posterior probability distribution
    is updated every time the agent gets assigned the role of hearer)

    :param population: a population (1D numpy array), where each agent is simply a LOG posterior probability
    distribution
    :param rounds: the number of rounds for which the population should communicate
    :return: the data that was produced during the communication rounds, as a list of (topic, utterance) tuples
    """
    if n_parents == 'single':
        if len(population) != 2 or interaction != 'taking_turns':
            raise ValueError(
                "OOPS! n_parents = 'single' only works if popsize = 2 and interaction = 'taking_turns'.")
        random_parent_index = np.random.choice(np.arange(len(population)))
    data = []
    for i in range(rounds):
        if interaction == 'taking_turns':
            if len(population) != 2:
                raise ValueError(
                "OOPS! interaction = 'taking_turns' only works if popsize = 2.")
            if i % 2 == 0:
                speaker_index = 0
                hearer_index = 1
            else:
                speaker_index = 1
                hearer_index = 0
        else:
            pair_indices = np.random.choice(np.arange(len(population)), size=2, replace=False)
            speaker_index = pair_indices[0]
            hearer_index = pair_indices[1]
        topic = random.choice(meanings)
        if mutual_understanding:
            speaker_language = sample(population[speaker_index])
            hearer_language = sample(population[hearer_index])
            if production == 'simlang':
                utterance = produce_simlang(speaker_language, topic)
            else:
                utterance = produce(speaker_language, topic, gamma, error, noise)  # whenever a speaker is called upon
            # to produce a utterance, they first sample a language from their posterior probability distribution. So
            # each agent keeps updating their language according to the data received from their communication partner.
            listener_response = receive_with_repair(hearer_language, utterance)
            counter = 0
            while '?' in listener_response:
                if counter == 3:  # After 3 attempts, the listener stops trying to do repair
                    break
                if production == 'simlang':
                    utterance = produce_simlang(speaker_language, topic)
                else:
                    utterance = produce(speaker_language, topic, gamma, error, noise_switch=False)  # For now, we assume
                                # that the speaker's response to a repair initiator always comes through without noise.
                listener_response = receive_with_repair(hearer_language, utterance)
                counter += 1
            if production == 'simlang':
                if observed_meaning == 'intended':
                    population[hearer_index] = update_posterior_simlang(population[hearer_index], topic,
                                                                    utterance)  # (Thus, in this simplified version of
                # the model, agents are still able to "track changes in their partners' linguistic behaviour over time
                elif observed_meaning == 'inferred':
                    population[hearer_index] = update_posterior_simlang(population[hearer_index], listener_response,
                                                                        utterance)  # (Thus, in this simplified version of
                    # the model, agents are still able to "track changes in their partners' linguistic behaviour over time
            else:
                if observed_meaning == 'intended':
                    population[hearer_index] = update_posterior(population[hearer_index], topic, utterance)
                elif observed_meaning == 'inferred':
                    population[hearer_index] = update_posterior(population[hearer_index], listener_response, utterance)

        else:
            if production == 'simlang':
                utterance = produce_simlang(sample(population[speaker_index]), topic)
            else:
                utterance = produce(sample(population[speaker_index]), topic, gamma, error, noise)  # whenever a speaker is
                # called upon to produce a utterance, they first sample a language from their posterior probability
                # distribution. So each agent keeps updating their language according to the data they receive from
                # their communication partner.
            if production == 'simlang':
                if observed_meaning == 'intended':
                    population[hearer_index] = update_posterior_simlang(population[hearer_index], topic, utterance) #(Thus,
                    # in this simplified version of the model, agents are still able to "track changes in their partners'
                    # linguistic behaviour over time
                elif observed_meaning == 'inferred':
                    hearer_language = sample(population[hearer_index])
                    inferred_meaning = receive_without_repair(hearer_language, utterance)
                    population[hearer_index] = update_posterior_simlang(population[hearer_index], inferred_meaning,
                                                                        utterance)  # (Thus,
                    # in this simplified version of the model, agents are still able to "track changes in their partners'
                    # linguistic behaviour over time
            else:
                if observed_meaning == 'intended':
                    population[hearer_index] = update_posterior(population[hearer_index], topic, utterance)
                elif observed_meaning == 'inferred':
                    hearer_language = sample(population[hearer_index])
                    inferred_meaning = receive_without_repair(hearer_language, utterance)
                    population[hearer_index] = update_posterior(population[hearer_index], inferred_meaning, utterance)

        if n_parents == 'single':

            if speaker_index == random_parent_index:
                if observed_meaning == 'intended':
                    data.append((topic, utterance))
                elif observed_meaning == 'inferred':
                    if mutual_understanding:
                        inferred_meaning = listener_response
                    data.append((inferred_meaning, utterance))
        else:
            if observed_meaning == 'intended':
                data.append((topic, utterance))
            elif observed_meaning == 'inferred':
                if mutual_understanding:
                    inferred_meaning = listener_response
                data.append((inferred_meaning, utterance))
    return data



def dataset_from_language(language):
    """
    Takes a language and generates a balanced minimal dataset from it, in which each possible meaning occurs exactly
    once, combined with its corresponding form.

    :param language: a language (list of forms_without_noisy_variants that has same length as the global variable
    meanings, where each form is mapped to the meaning at the corresponding index)
    :return: a dataset (list containing tuples, where each tuple is a meaning-form pair, with the meaning followed by
    the form)
    """
    meaning_form_pairs = []
    for i in range(len(language)):
        meaning = meanings[i]
        form = language[i]
        meaning_form_pairs.append((meaning, form))
    return meaning_form_pairs


def create_initial_dataset(desired_class, b):
    """
    Creates a balanced dataset from a randomly chosen language of the desired class.

    :param desired_class: 'degenerate', 'holistic', 'hybrid', 'compositional', or 'other'; category indices as hardcoded
    in classify_language function are: 0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, 4 = other (here I'm
    following the ordering used in the Kirby et al., 2015 paper; NOT the ordering from SimLang lab 21)
    :return: a dataset (list containing tuples, where each tuple is a meaning-form pair, with the meaning followed by
    the form) from a randomly chosen language of the desired class
    """
    if desired_class == 'degenerate':
        class_index = 0
    elif desired_class == 'holistic':
        class_index = 1
    elif desired_class == 'hybrid':
        class_index = 2
    elif desired_class == 'compositional':
        class_index = 3
    elif desired_class == 'other':
        class_index = 4
    language_class_indices = np.where(class_per_lang == class_index)[0]
    class_languages = []
    for index in language_class_indices:
        class_languages.append(all_possible_languages[index])
    random_language = random.choice(class_languages)
    meaning_form_pairs = dataset_from_language(random_language)
    if b % len(meaning_form_pairs) != 0:
        raise ValueError("OOPS! b needs to be a multiple of the number of meanings in order for this function to create a balanced dataset.")
    dataset = []
    for i in range(int(b/len(meaning_form_pairs))):
        dataset = dataset+meaning_form_pairs
    return dataset



def language_stats(population):
    """
    Tracks how well each of the language classes is represented in the populations' posterior probability distributions

    :param population: a population (1D numpy array), where each agent is simply a LOG posterior probability
    distribution
    :return: a list containing the overall average posterior probability assigned to each class of language in the
    population, where index 0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, 4 = other; these are the
    category indices as hardcoded in the classify_language() function (where I follow the ordering used in the Kirby
    et al., 2015 paper; NOT the ordering from SimLang lab 21)
    """
    stats = np.zeros(5)  # 0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, 4 = other
    for p in population:
        for i in range(len(p)):
            # if proportion_measure == 'posterior':
                # stats[int(class_per_lang[i])] += np.exp(p[i]) / len(population)
            stats[int(class_per_lang[i])] += np.exp(p[i])
            # elif proportion_measure == 'sampled':
            #     sampled_lang_index = log_roulette_wheel(p)
            #     stats[int(class_per_lang[sampled_lang_index])] += 1
    stats = np.divide(stats, len(population))
    return stats



def simulation(generations, rounds, bottleneck, popsize, data):
    """
    Runs the full simulation and returns the total amount of posterior probability that is assigned to each language
    class over generations (results) as well as the data that each generation produced (data)

    :param generations: the desired number of generations (int)
    :param rounds: the desired number of communication rounds *within* each generation
    :param bottleneck: the amount of data (<meaning, form> pairs) that each learner receives
    :param popsize: the desired size of the population (int)
    :param data: the initial data that generation 0 learns from
    :return:
    """

    results = []
    population = new_population(popsize)
    for i in range(generations):
        for j in range(popsize):
            for k in range(bottleneck):
                if interaction == 'taking_turns':
                    if len(population) != 2:
                        raise ValueError(
                            "OOPS! interaction = 'taking_turns' only works if popsize = 2.")
                    if bottleneck != len(data):
                        raise ValueError(
                            "UH-OH! data should have the same size as the bottleneck b")
                    meaning, signal = data[k]
                else:
                    meaning, signal = random.choice(data)
                if production == 'simlang':
                    population[j] = update_posterior_simlang(population[j], meaning, signal)
                else:
                    population[j] = update_posterior(population[j], meaning, signal)
        data = population_communication(population, rounds)
        results.append(language_stats(population))
        if turnover:
            population = new_population(popsize)
    return results, data



def str_to_bool(s):
    """
    Takes a string which is either 'True' or 'False' and turns it into a boolean True or False

    :param s:
    :return:
    """
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError  # evil ValueError that doesn't tell you what the wrong value was



turnover = True  # determines whether new individuals enter the population or not
b = 20  # the bottleneck (i.e. number of meaning-form pairs the each pair gets to see during training (Kirby et al.
        # used a bottleneck of 20 in the body of the paper.
rounds = 2*b  # Kirby et al. (2015) used rounds = 2*b, but SimLang lab 21 uses 1*b
popsize = 2  # If I understand it correctly, Kirby et al. (2015) used a population size of 2: each generation is simply
            # a pair of agents.
runs = 50  # the number of independent simulation runs (Kirby et al., 2015 used 100)
generations = 100  # the number of generations (Kirby et al., 2015 used 100)
initial_language_type = 'degenerate'  # set the language class that the first generation is trained on

noise = True  # parameter that determines whether environmental noise is on or off
noise_prob = float(sys.argv[1])  # Setting the 'noise_prob' parameter based on the command-line input #NOTE: first argument in sys.argv list is always the name of the script  # the probability of environmental noise masking part of an utterance

print("sys.argv[1] is:")
print(sys.argv[1])
print("type(sys.argv[1]) is:")
print(type(sys.argv[1]))

print("noise_prob is:")
print(noise_prob)
print("type(noise_prob) is:")
print(type(noise_prob))



# proportion_measure = 'posterior'  # the way in which the proportion of language classes present in the population is
# measured. Can be set to either 'posterior' (where we directly measure the total amount of posterior probability
# assigned to each language class), or 'sampled' (where at each generation we make all agents in the population pick a
# language and we count the resulting proportions.
production = 'my_code'  # can be set to 'simlang' or 'my_code'
mutual_understanding = str_to_bool(sys.argv[2])  # Setting the 'mutual_understanding' parameter based on the command-line input #NOTE: first argument in sys.argv list is always the name of the script

print("sys.argv[2] is:")
print(sys.argv[2])
print("str_to_bool(sys.argv[2]) is:")
print(str_to_bool(sys.argv[2]))
print("type(str_to_bool(sys.argv[2])) is:")
print(type(str_to_bool(sys.argv[2])))

if mutual_understanding:
    gamma = 2  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
    # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
    # conditions
else:
    gamma = 0  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
    # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
    # conditions
minimal_effort = str_to_bool(sys.argv[3])  # Setting the 'minimal_effort' parameter based on the command-line input #NOTE: first argument in sys.argv list is always the name of the script

print("sys.argv[3] is:")
print(sys.argv[3])
print("str_to_bool(sys.argv[3]) is:")
print(str_to_bool(sys.argv[3]))
print("type(str_to_bool(sys.argv[3])) is:")
print(type(str_to_bool(sys.argv[3])))


cost_vector = [0.0, 0.15, 0.45]  # costs of no repair, restricted request, and open request, respectively
compressibility_bias = False  # determines whether agents have a prior that favours compressibility, or a flat prior
observed_meaning = 'intended'  # determines which meaning the learner observes when receiving a meaning-form pair; can
# be set to either 'intended', where the learner has direct access to the speaker's intended meaning, or 'inferred',
# where the learner has access to the hearer's interpretation.
interaction = 'taking_turns'  # can be set to either 'random' or 'taking_turns'. The latter is what Kirby et al. (2015)
# used, but NOTE that it only works with a popsize of 2!
n_parents = 'single'  # determines whether each generation of learners receives data from a single agent from the
# previous generation, or from multiple (can be set to either 'single' or 'multiple').

gen_start = int(generations/2)

n_lang_classes = 5  # the number of language classes that are distinguished (int). This should be 4 if the old code was
# used (from before 13 September 2019, 1:30 pm), which did not yet distinguish between 'holistic' and 'hybrid'
# languages, and 5 if the new code was used which does make this distinction.



if __name__ == '__main__':

    if compressibility_bias:
        priors = new_log_prior
    else:
        priors = np.ones(len(all_possible_languages))
        priors = np.divide(priors, np.sum(priors))
        priors = np.log(priors)

    initial_dataset = create_initial_dataset(initial_language_type, b)  # the data that the first generation learns from


    results = []
    for i in range(runs):
        results.append(simulation(generations, rounds, b, popsize, initial_dataset)[0])

    timestr = time.strftime("%Y%m%d-%H%M%S")

    pickle_file_title = "Pickle_r_" + str(runs) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_pop_size_" + str(popsize) + "_mutual_u_"+str(mutual_understanding)+ "_gamma_" + str(gamma) +"_minimal_e_"+str(minimal_effort)+ "_c_"+str(cost_vector)+ "_turnover_" + str(turnover) + "_bias_" +str(compressibility_bias) + "_init_" + initial_language_type + "_noise_" + str(noise) + "_noise_prob_" + str(noise_prob)+"_"+production+"_observed_m_"+observed_meaning+"_n_lang_classes_"+str(n_lang_classes)+"_"+timestr

    pickle.dump(results, open(pickle_file_title+".p", "wb"))

