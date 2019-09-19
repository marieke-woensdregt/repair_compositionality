import numpy as np
import itertools
import random
from copy import deepcopy
from math import log
import pickle
import time

if __name__ == '__main__':
    import scipy.special
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

###################################################################################################################
# ALL PARAMETER SETTINGS GO HERE:

# FROM SIMLANG LAB 21:
languages_simlang = [[('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')]]
types_simlang = [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]
priors_simlang = [-0.9178860550328204, -10.749415928290118, -10.749415928290118, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -12.460704095246543, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -12.460704095246543, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -2.304180416152711, -11.272664072079987, -10.749415928290118, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -10.749415928290118, -11.272664072079987, -2.304180416152711, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -12.460704095246543, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -12.460704095246543, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -10.749415928290118, -10.749415928290118, -0.9178860550328204]

# MY OWN CODE:
meanings = ['02', '03', '12', '13']  # all possible meanings
forms_without_noise = ['aa', 'ab', 'ba', 'bb']  # all possible forms, excluding their possible 'noisy variants'
noisy_forms = ['a_', 'b_', '_a', '_b']  # all possible noisy variants of the forms above
all_forms_including_noisy_variants = forms_without_noise+noisy_forms  # all possible forms, including both complete
# forms and noisy variants
error = 0.05  # the probability of making a production error (Kirby et al., 2015 use 0.05)

turnover = True  # determines whether new individuals enter the population or not
b = 20  # the bottleneck (i.e. number of meaning-form pairs the each pair gets to see during training (Kirby et al.
        # used a bottleneck of 20 in the body of the paper.
rounds = 2*b  # Kirby et al. (2015) used rounds = 2*b, but SimLang lab 21 uses 1*b
popsize = 2  # If I understand it correctly, Kirby et al. (2015) used a population size of 2: each generation is simply
            # a pair of agents.
runs = 10  # the number of independent simulation runs (Kirby et al., 2015 used 100)
generations = 15  # the number of generations (Kirby et al., 2015 used 100)
initial_language_type = 'degenerate'  # set the language class that the first generation is trained on

production = 'my_code'  # can be set to 'simlang' or 'my_code'

cost_vector = np.array([0.0, 0.2, 0.4])  # costs of no repair, restricted request, and open request, respectively
compressibility_bias = False  # determines whether agents have a prior that favours compressibility, or a flat prior
observed_meaning = 'intended'  # determines which meaning the learner observes when receiving a meaning-form pair; can
# be set to either 'intended', where the learner has direct access to the speaker's intended meaning, or 'inferred',
# where the learner has access to the hearer's interpretation.
interaction = 'taking_turns'  # can be set to either 'random' or 'taking_turns'. The latter is what Kirby et al. (2015)
# used, but NOTE that it only works with a popsize of 2!
n_parents = 'single'  # determines whether each generation of learners receives data from a single agent from the
# previous generation, or from multiple (can be set to either 'single' or 'multiple').

# proportion_measure = 'posterior'  # the way in which the proportion of language classes present in the population is
# measured. Can be set to either 'posterior' (where we directly measure the total amount of posterior probability
# assigned to each language class), or 'sampled' (where at each generation we make all agents in the population pick a
# language and we count the resulting proportions.

burn_in = round(generations / 2)  # the burn-in period that is excluded when calculating the mean distribution over languages after convergence

n_lang_classes = 5  # the number of language classes that are distinguished (int). This should be 4 if the old code was
# used (from before 13 September 2019, 1:30 pm), which did not yet distinguish between 'holistic' and 'hybrid'
# languages, and 5 if the new code was used which does make this distinction.


# AND SOME MORE PARAMETER SETTINGS THAT SHOULD BE SPECIFIC TO THIS PARTICULAR SIMULATION RUN (SO NOT IMPORTED):
if __name__ == '__main__':

    noise = True  # parameter that determines whether environmental noise is on or off
    noise_prob = 0.6  # the probability of environmental noise masking part of an utterance

    mutual_understanding = True
    if mutual_understanding:
        gamma = 2  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
        # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
        # conditions
    else:
        gamma = 0  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
        # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
        # conditions

    minimal_effort = True

    communicative_success_pressure = False  # determines whether there is a pressure for communicative success or not
    communicative_success_pressure_strength = (2./3.)  # determines how much more likely a <meaning, form> pair from a
    # successful interaction is to enter the data set that is passed on to the next generation, compared to a
    # <meaning, form> pair from a unsuccessful interaction.

    pickle_file_path = "pickles/"

    fig_file_path = "plots/"


###################################################################################################################
# FIRST SOME FUNCTIONS TO CREATE ALL POSSIBLE LANGUAGES AND CLASSIFY THEM:

def create_all_possible_languages(meaning_list, forms):
    """Creates all possible languages

    :param meaning_list: list of strings corresponding to all possible meanings
    :param forms: list of strings corresponding to all possible forms_without_noisy_variants
    :returns: list of tuples which represent languages, where each tuple consists of forms_without_noisy_variants and
    has length len(meanings)
    """
    all_possible_languages = list(itertools.product(forms, repeat=len(meaning_list)))
    return all_possible_languages


def classify_language(lang, forms, meaning_list):
    """
    Classify one particular language as either 'degenerate' (0), 'holistic' (1), 'other' (2)
    or 'compositional' (3) (Kirby et al., 2015)

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :param forms: list of strings corresponding to all possible forms_without_noisy_variants
    :param meaning_list: list of strings corresponding to all possible meanings
    :returns: integer corresponding to category that language belongs to:
    0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, 4 = other (here I'm following the
    ordering used in the Kirby et al., 2015 paper; NOT the ordering from SimLang lab 21)
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
    if len(lang) != len(meaning_list):
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


def classify_all_languages(language_list, complete_forms, meaning_list):
    """
    Classify all languages as either 'degenerate' (0), 'holistic' (1), 'other' (2) or 'compositional' (3)
    (Kirby et al., 2015)

    :param language_list: list of all languages
    :param complete_forms: list containing all possible complete forms; corresponds to global variable
    'forms_without_noise'
    :param meanings: list of all possible meanings; corresponds to global variable 'meanings'
    :returns: 1D numpy array containing integer corresponding to category of corresponding
    language index as hardcoded in classify_language function: 0 = degenerate, 1 = holistic, 2 = hybrid,
    3 = compositional, 4 = other (here I'm following the ordering used in the Kirby et al., 2015 paper; NOT the ordering
    from SimLang lab 21)
    """
    class_per_lang = np.zeros(len(language_list))
    for l in range(len(language_list)):
        class_per_lang[l] = classify_language(language_list[l], complete_forms, meaning_list)
    return class_per_lang


# NOW SOME FUNCTIONS TO CHECK MY CODE FOR CREATING AND CLASSIFYING ALL LANGUAGES AGAINST THE LISTS OF LANGUAGES AND
# TYPES THAT WERE COPIED INTO LAB 21 OF THE SIMLANG COURSE:

def transform_all_languages_to_simlang_format(language_list, meaning_list):
    """
    Takes a list of languages as represented by me (with only the forms_without_noisy_variants listed
    for each language, assuming the meaning for each form is specified by the
    form's index), and turning it into a list of languages as represented in
    SimLang lab 21 (which in turn is based on Kirby et al., 2015), in which a
    <meaning, form> pair forms_without_noisy_variants a tuple, and four of those tuples in a list form
    a language

    :param language_list: list of all languages
    :param meaning_list: list of all possible meanings; corresponds to global variable 'meanings'
    :returns: list of the input languages in the format of SimLang lab 21
    """
    all_langs_as_in_simlang = []
    for l in range(len(language_list)):
        lang_as_in_simlang = [(meaning_list[x], language_list[l][x]) for x in range(len(meaning_list))]
        all_langs_as_in_simlang.append(lang_as_in_simlang)
    return all_langs_as_in_simlang


def check_all_lang_lists_against_each_other(language_list_a, language_list_b, priors_simlang):
    """
    Takes two lists of languages of the same length and format, and checks for each languages in language_list_a,
    whether it is also present in language_list_b.

    :param language_list_a: list of languages represented as in the SimLang lab 21 code, where each language is a list
    of 4 tuples, where each tuple consists of a meaning and its corresponding form.
    :param language_list_b: list of languages of same format as language_list_b
    :param priors_simlang: list of LOG priors copied from the SimLang lab 21 notebook
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


# NOW SOME FUNCTIONS THAT HANDLE PRODUCTION, NOISY PRODUCTION, AND RECEPTION WITH AND WITHOUT REPAIR:

# A reproduction of the production function of Kirby et al. (2015):
def production_likelihoods_kirby_et_al(language, topic, ambiguity_penalty, error_prob):
    """
    Calculates the production probabilities for each of the possible forms_without_noisy_variants given a language and
    topic, as defined by Kirby et al. (2015)

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param ambiguity_penalty: parameter that determines the strength of the penalty on ambiguity (gamma)
    :param error_prob: the probability of making an error in production
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
    prop_to_prob_correct_form = ((1./ambiguity) ** ambiguity_penalty) * (1. - error_prob)
    prop_to_prob_error_form = error_prob / (len(forms_without_noise) - 1)
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
    while i < len(my_list):
        if my_list[i] == element_to_be_removed:
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


def production_likelihoods_with_noise(language, topic, ambiguity_penalty, error_prob, prob_of_noise):
    """
    Calculates the production probabilities for each of the possible forms (including both forms without noise and all
    possible noisy variants) given a language and topic, and the probability of environmental noise

    :param language: list of forms that has same length as list of meanings (global variable), where each form is
    mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param ambiguity_penalty: parameter that determines the strength of the penalty on ambiguity (gamma)
    :param error_prob: the probability of making an error in production
    :param prob_of_noise: the probability of environmental noise masking part of the utterance
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
    prop_to_prob_correct_form_complete = ((1./ambiguity) ** ambiguity_penalty) * (1. - error_prob) * (1 - prob_of_noise)
    prop_to_prob_error_form_complete = error_prob / (len(forms_without_noise) - 1) * (1 - prob_of_noise)
    prop_to_prob_correct_form_noisy = ((1. / ambiguity) ** ambiguity_penalty) * (1. - error_prob) * (prob_of_noise / len(noisy_forms))
    prop_to_prob_error_form_noisy = error_prob / (len(forms_without_noise) - 1) * (1 - prob_of_noise) * (prob_of_noise / len(noisy_forms))
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


# And finally, let's write a function that actually produces an utterance, given a language and a topic:
def produce(language, topic, ambiguity_penalty, error_prob, noise_switch):
    """
    Produces an actual utterance, given a language and a topic

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param ambiguity_penalty: parameter that determines the strength of the penalty on ambiguity (gamma)
    :param error_prob: the probability of making an error in production
    :param noise_switch: turns noise on when set to True, and off when set to False
    :return: an utterance. That is, a single form chosen from either the global variable "forms_without_noise" (if
    noise is False) or the global variable "all_forms_including_noisy_variants" (if noise is True).
        """
    if noise_switch:
        prop_to_prob_per_form_array = production_likelihoods_with_noise(language, topic, ambiguity_penalty, error_prob, noise_prob)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(all_forms_including_noisy_variants, p=prob_per_form_array)
    else:
        prop_to_prob_per_form_array = production_likelihoods_kirby_et_al(language, topic, ambiguity_penalty, error_prob)
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


def noisy_to_complete_forms(noisy_form, complete_forms):
    """
    Takes a noisy form and returns all possible complete forms that it's compatible with.

    :param noisy_form: a noisy form (i.e. a string containing '_' as at least one of the characters)
    :param complete_forms: The full set of possible complete forms (corresponds to global parameter
    'forms_without_noise')
    :return: A list of complete forms that the noisy form is compatible with
    """
    possible_complete_forms = []
    amount_of_noise = noisy_form.count('_')
    for complete_form in complete_forms:
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
# probabilities for the different response options and putting that in a separate function?
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


# AND NOW FOR THE FUNCTIONS THAT DO THE BAYESIAN LEARNING:

if __name__ == '__main__':  # I'm defining this function separately in this module and in the module for ponyland,
# because ponyland seems to have an older version of scipy installed in which scipy.special didn't include the
# logsumexp() function yet.
    def update_posterior(log_posterior, hypotheses, topic, utterance, ambiguity_penalty, noise_switch, prob_of_noise, all_possible_forms):
        """
        Takes a LOG posterior probability distribution and a <topic, utterance> pair, and updates the posterior probability
        distribution accordingly

        :param log_posterior: 1D numpy array containing LOG posterior probability values for each hypothesis
        :param hypotheses: list of all possible languages
        :param topic: a topic (string from the global variable meanings)
        :param utterance: an utterance (string from the global variable forms (can be a noisy form if parameter noise is
        True)
        :param ambiguity_penalty: parameter that determines extent to which speaker tries to avoid ambiguity; corresponds
        to global variable 'gamma'
        :param noise_switch: determines whether noise is on or off (set to either True or False); corresponds to global
        variable 'noise'
        :param prob_of_noise: the probability of noise (only relevant when noise_switch == True); corresponds to global
        variable 'noise_prob'
        :param all_possible_forms: list of all possible forms INCLUDING noisy variants; corresponds to global variable
        'all_forms_including_noisy_variants'
        :return: the updated (and normalized) log_posterior (1D numpy array)
        """
        # First, let's find out what the index of the utterance is in the list of all possible forms (including the noisy
        # variants):
        for i in range(len(all_possible_forms)):
            if all_possible_forms[i] == utterance:
                utterance_index = i
        # Now, let's go through each hypothesis (i.e. language), and update its posterior probability given the
        # <topic, utterance> pair that was given as input:
        new_log_posterior = []
        for j in range(len(log_posterior)):
            hypothesis = hypotheses[j]
            if noise_switch:
                likelihood_per_form_array = production_likelihoods_with_noise(hypothesis, topic, ambiguity_penalty, error, prob_of_noise)
            else:
                likelihood_per_form_array = production_likelihoods_kirby_et_al(hypothesis, topic, ambiguity_penalty, error)
            log_likelihood_per_form_array = np.log(likelihood_per_form_array)
            new_log_posterior.append(log_posterior[j] + log_likelihood_per_form_array[utterance_index])

        new_log_posterior_normalized = np.subtract(new_log_posterior, scipy.special.logsumexp(new_log_posterior))

        return new_log_posterior_normalized


def normalize_logprobs_simlang(logprobs):
    """
    This function is copied directly from lab 21 of the SimLang course of 2019.

    :param logprobs: a list of LOG probabilities
    :return: a list of normalised LOG probabilities
    """
    logtotal = scipy.special.logsumexp(logprobs) #calculates the summed log probabilities
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


# AND NOW FOR THE FUNCTIONS THAT HANDLE SAMPLING A HYPOTHESIS FROM A POSTERIOR PROBABILITY DISTRIBUTION:

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
        accumulator = scipy.special.logsumexp([accumulator, normedlogs[i + 1]])


def sample(log_posterior):
    """
    Samples a language based on the posterior

    :param log_posterior: a list of LOG posterior probabilities
    :return: a language (list of forms_without_noisy_variants that has same length as the global variable meanings,
    where each form is mapped to the meaning at the corresponding index)
    """
    return hypothesis_space[log_roulette_wheel(log_posterior)]


# NOW THE FUNCTION THAT CREATE A NEW POPULATION, AND MAKES A POPULATION COMMUNICATE (FOR THE INTRA-GENERATIONAL
# INTERACTION ROUNDS):

def new_population(pop_size, log_priors):
    """
    Creates a new population of agents, where each agent simply consists of the prior probability distribution (which
    is assumed to be defined as a global variable called 'priors')

    :param pop_size: the number of agents desired in the new population; corresponds to global variable 'popsize'
    :param log_priors: the LOG prior probability distribution that each agent should be initialised with
    :return: 2D numpy array, with agents on the rows, and hypotheses (or rather their corresponding LOG prior
    probabilities)
    on the columns.
    """
    population = [log_priors for x in range(pop_size)]
    population = np.array(population)
    return population


# TODO: population_communication() has become a bit of a monster function; see if I can take some parts out to be
#  separate functions (e.g. for the different possible settings of the mutual_understanding and minimal_effort
#  parameters. Also, there has to be a way to not have exactly the same lines of code for doing the communicative
#  success pressure stuff in there twice (should probably be a separate function)!
def population_communication(population, n_rounds):
    """
    Takes a population, makes it communicate for a number of rounds (where agents' posterior probability distribution
    is updated every time the agent gets assigned the role of hearer)

    :param population: a population (1D numpy array), where each agent is simply a LOG posterior probability
    distribution
    :param n_rounds: the number of rounds for which the population should communicate; corresponds to global variable
    'rounds'
    :return: the data that was produced during the communication rounds, as a list of (topic, utterance) tuples
    """
    if n_parents == 'single':
        if len(population) != 2 or interaction != 'taking_turns':
            raise ValueError(
                "OOPS! n_parents = 'single' only works if popsize = 2 and interaction = 'taking_turns'.")
        random_parent_index = np.random.choice(np.arange(len(population)))
    data = []
    data_for_just_in_case = []
    for i in range(n_rounds):
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
        speaker_language = sample(population[speaker_index])
        hearer_language = sample(population[hearer_index])
        if mutual_understanding is True:
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
                    population[hearer_index] = update_posterior(population[hearer_index], hypothesis_space, topic, utterance, gamma, noise, noise_prob, all_forms_including_noisy_variants)
                elif observed_meaning == 'inferred':
                    population[hearer_index] = update_posterior(population[hearer_index], hypothesis_space, listener_response, utterance, gamma, noise, noise_prob, all_forms_including_noisy_variants)

        elif mutual_understanding is False:
            if production == 'simlang':
                utterance = produce_simlang(speaker_language, topic)
            else:
                utterance = produce(speaker_language, topic, gamma, error, noise)  # whenever a speaker is
                # called upon to produce a utterance, they first sample a language from their posterior probability
                # distribution. So each agent keeps updating their language according to the data they receive from
                # their communication partner.
            if production == 'simlang':
                if observed_meaning == 'intended':
                    population[hearer_index] = update_posterior_simlang(population[hearer_index], topic, utterance) #(Thus,
                    # in this simplified version of the model, agents are still able to "track changes in their partners'
                    # linguistic behaviour over time
                elif observed_meaning == 'inferred':
                    inferred_meaning = receive_without_repair(hearer_language, utterance)
                    population[hearer_index] = update_posterior_simlang(population[hearer_index], inferred_meaning,
                                                                        utterance)  # (Thus,
                    # in this simplified version of the model, agents are still able to "track changes in their partners'
                    # linguistic behaviour over time
            else:
                if observed_meaning == 'intended':
                    population[hearer_index] = update_posterior(population[hearer_index], hypothesis_space, topic, utterance, gamma, noise, noise_prob, all_forms_including_noisy_variants)
                elif observed_meaning == 'inferred':
                    inferred_meaning = receive_without_repair(hearer_language, utterance)
                    population[hearer_index] = update_posterior(population[hearer_index], hypothesis_space, inferred_meaning, utterance, gamma, noise, noise_prob, all_forms_including_noisy_variants)

        if n_parents == 'single':

            if speaker_index == random_parent_index:
                if observed_meaning == 'intended':
                    meaning_observed = topic
                    inferred_meaning = receive_without_repair(hearer_language, utterance)

                elif observed_meaning == 'inferred':
                    if mutual_understanding:
                        meaning_observed = listener_response
                    else:
                        meaning_observed = inferred_meaning

                if communicative_success_pressure is True:
                    if mutual_understanding:
                        inferred_meaning = listener_response
                    success = False
                    if inferred_meaning == topic:
                        success = True
                    random_float = np.random.uniform()
                    if random_float < communicative_success_pressure_strength:  # if our random_float falls within
                        # the range of our communicative_success_pressure_strength parameter, the <meaning, form>
                        # pair is added to the dataset only if the interaction was successful
                        if success:
                            data.append((meaning_observed, utterance))
                    else:  # if our random_float falls above the range of our
                        # communicative_success_pressure_strength parameter, the <meaning, form>
                        # pair is added to the dataset no matter whether the interaction was successful or not
                        data.append((meaning_observed, utterance))
                elif communicative_success_pressure is False:
                    data.append((meaning_observed, utterance))

                data_for_just_in_case.append((meaning_observed, utterance))  # this is being recorded just in case the
                # dataset otherwise ends up being empty, as a result of the pressure for communicative success being
                # too strong.


        elif n_parents == 'multiple':
            if observed_meaning == 'intended':
                meaning_observed = topic
                inferred_meaning = receive_without_repair(hearer_language, utterance)

            elif observed_meaning == 'inferred':
                if mutual_understanding:
                    meaning_observed = listener_response
                else:
                    meaning_observed = inferred_meaning

            if communicative_success_pressure is True:
                if mutual_understanding:
                    inferred_meaning = listener_response
                success = False
                if inferred_meaning == topic:
                    success = True
                random_float = np.random.uniform()
                if random_float < communicative_success_pressure_strength:  # if our random_float falls within
                    # the range of our communicative_success_pressure_strength parameter, the <meaning, form>
                    # pair is added to the dataset only if the interaction was successful
                    if success:
                        data.append((meaning_observed, utterance))
                else:  # if our random_float falls above the range of our
                    # communicative_success_pressure_strength parameter, the <meaning, form>
                    # pair is added to the dataset no matter whether the interaction was successful or not
                    data.append((meaning_observed, utterance))
            elif communicative_success_pressure is False:
                data.append((meaning_observed, utterance))

            data_for_just_in_case.append((meaning_observed, utterance))  # this is being recorded just in case the
                # dataset otherwise ends up being empty, as a result of the pressure for communicative success being
                # too strong.

    if len(data) == 0:  # In case the data set is empty (which might happen if the pressure for communicative success
        # is too strong; especially if the "n_parents" parameter has been set to 'single'), we just use all the
        # <meaning, form> pairs from all the interactions as the data set, no matter whether they were successful or
        # not.
        data = data_for_just_in_case


    # I ADDED THE BIT BELOW TO CHECK HOW OFTEN SELECTING FOR COMMUNICATIVE SUCCESS LEADS TO DATA SETS THAT ARE SMALLER
    # THAN THE BOTTLENECK:
    if communicative_success_pressure:
        if len(data) < b:
            print('')
            print('')
            print("UH-OH! len(data) < b!")
            print("len(data) is:")
            print(len(data))

    return data


# AND NOW FOR THE FUNCTIONS THAT HANDLE CREATING A DATASET FROM A SPECIFIC LANGUAGE TYPE (IN ORDER TO GENERATE THE DATA
# THAT GENERATION 0 WILL LEARN FROM):

def dataset_from_language(language, meaning_list):
    """
    Takes a language and generates a balanced minimal dataset from it, in which each possible meaning occurs exactly
    once, combined with its corresponding form.

    :param language: a language (list of forms_without_noisy_variants that has same length as the global variable
    meanings, where each form is mapped to the meaning at the corresponding index)
    :param meaning_list: list containing all possible meanings; corresponds to global variable 'meanings'
    :return: a dataset (list containing tuples, where each tuple is a meaning-form pair, with the meaning followed by
    the form)
    """
    meaning_form_pairs = []
    for i in range(len(language)):
        meaning = meaning_list[i]
        form = language[i]
        meaning_form_pairs.append((meaning, form))
    return meaning_form_pairs


def create_initial_dataset(desired_class, bottleneck, language_list, class_per_language, meaning_list):
    """
    Creates a balanced dataset from a randomly chosen language of the desired class.

    :param desired_class: 'degenerate', 'holistic', 'hybrid', 'compositional', or 'other'; category indices as hardcoded
    in classify_language function are: 0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, 4 = other (here I'm
    following the ordering used in the Kirby et al., 2015 paper; NOT the ordering from SimLang lab 21)
    :param bottleneck: the transmission bottleneck (int); corresponds to global variable 'b'
    :param language_list: list of all languages
    :param class_per_language: list of len(hypothesis_space) which contains an integer indicating the class of the
    language at the corresponding index in the global variable hypothesis_space
    :param meaning_list: list of all possible meanings; corresponds to global variable 'meanings'
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
    language_class_indices = np.where(class_per_language == class_index)[0]
    class_languages = []
    for index in language_class_indices:
        class_languages.append(language_list[index])
    random_language = random.choice(class_languages)
    meaning_form_pairs = dataset_from_language(random_language, meaning_list)
    if bottleneck % len(meaning_form_pairs) != 0:
        raise ValueError("OOPS! b needs to be a multiple of the number of meanings in order for this function to create a balanced dataset.")
    dataset = []
    for i in range(int(bottleneck / len(meaning_form_pairs))):
        dataset = dataset+meaning_form_pairs
    return dataset


# AND NOW A FUNCTION THAT RECORDS THE PROPORTION OF POSTERIOR PROBABILITY THAT IS ASSIGNED TO EACH LANGUAGE CLASS IN A
# GIVEN GENERATION:

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
    stats = np.zeros(5)
    for p in population:
        for i in range(len(p)):
            # if proportion_measure == 'posterior':  # Note that this will only work when the population has a size
            ## that is a reasonable multitude of the number of language classes
                # stats[int(class_per_lang[i])] += np.exp(p[i]) / len(population)
            stats[int(class_per_lang[i])] += np.exp(p[i])
            # elif proportion_measure == 'sampled':
            #     sampled_lang_index = log_roulette_wheel(p)
            #     stats[int(class_per_lang[sampled_lang_index])] += 1
    stats = np.divide(stats, len(population))
    return stats


# AND NOW FINALLY FOR THE FUNCTION THAT RUNS THE ACTUAL SIMULATION:

if __name__ == '__main__':  # I'm defining this function separately in this module and in the module for ponyland,
# because ponyland seems to have an older version of scipy installed in which scipy.special didn't include the
# logsumexp() function yet, and because the simulation() function below calls the update_posterior() function defined
# above, which itself in turn calls the logsumexp() function.
    def simulation(n_gens, n_rounds, bottleneck, pop_size, hypotheses, log_priors, data, interaction_order, production_implementation, ambiguity_penalty, noise_switch, prob_of_noise, all_possible_forms):
        """
        Runs the full simulation and returns the total amount of posterior probability that is assigned to each language
        class over generations (language_stats_over_gens) as well as the data that each generation produced (data)

        :param n_gens: the desired number of generations (int); corresponds to global variable 'generations'
        :param n_rounds: the desired number of communication rounds *within* each generation; corresponds to global variable
        'rounds'
        :param bottleneck: the amount of data (<meaning, form> pairs) that each learner receives
        :param pop_size: the desired size of the population (int); corresponds to global variable 'popsize'
        :param hypotheses: list of all possible languages; corresponds to global variable 'hypothesis_space'
        :param log_priors: the LOG prior probability distribution that each agent should be initialised with
        :param data: the initial data that generation 0 learns from
        :param interaction_order: the order in which agents take turns in interaction (can be set to either 'taking_turns'
        or 'random')
        :param production_implementation: can be set to either 'my_code' or 'simlang'
        :param ambiguity_penalty: parameter that determines the extent to which the speaker tries to avoid ambiguity;
        corresponds to global variable 'gamma'
        :param noise_switch: determines whether noise is on or off (set to either True or False); corresponds to global
        variable 'noise'
        :param prob_of_noise: probability of noise (only relevant when noise_switch == True); corresponds to global variable
        'noise_prob'
        :param all_possible_forms: list of all possible forms INCLUDING noisy variants; corresponds to global variable
        'all_forms_including_noisy_variants'
        :return: language_stats_over_gens (which contains language stats over generations over runs), data (which contains
        data over generations over runs), and the final population
        """
        language_stats_over_gens = []
        data_over_gens = []
        population = new_population(pop_size, log_priors)
        for i in range(n_gens):
            for j in range(pop_size):
                for k in range(bottleneck):
                    if interaction_order == 'taking_turns':
                        if len(population) != 2:
                            raise ValueError(
                                "OOPS! interaction = 'taking_turns' only works if popsize = 2.")
                        if bottleneck != len(data):
                            raise ValueError(
                                "UH-OH! data should have the same size as the bottleneck b")
                        meaning, signal = data[k]
                    else:
                        meaning, signal = random.choice(data)
                    if production_implementation == 'simlang':
                        population[j] = update_posterior_simlang(population[j], meaning, signal)
                    else:
                        population[j] = update_posterior(population[j], hypotheses, meaning, signal, ambiguity_penalty, noise_switch, prob_of_noise, all_possible_forms)
            data = population_communication(population, n_rounds)
            language_stats_over_gens.append(language_stats(population))
            data_over_gens.append(data)
            if turnover:
                population = new_population(pop_size, priors)
        return language_stats_over_gens, data_over_gens, population


# AND NOW SOME FUNCTIONS THAT CONVERT A DATA LIST INTO A PANDAS DATA FRAME FOR PLOTTING:

def language_stats_to_dataframe(results, n_runs, n_gens, n_language_classes):
    """
    Takes a results list and puts it in a pandas dataframe together with other relevant variables (runs, generations,
    and language class)

    :param results: a list containing proportions for each of the 5 language classes, for each generation, for each run
    :param n_runs: the number of runs (int); corresponds to global variable 'runs'
    :param n_gens: the number of generations (int); corresponds to global variable 'generations'
    :param n_language_classes: the number of language classes that are distinguished (int); corresponds to global
    variable 'n_lang_classes'. This should be 4 if the old code was used (from before 13 September 2019, 1:30 pm), which
    did not yet distinguish between 'holistic' and 'hybrid' languages, and 5 if the new code was used which does make
    this distinction
    :return: a pandas dataframe containing four columns: 'run', 'generation', 'proportion' and 'class'
    """
    column_proportion = np.array(results)
    column_proportion = column_proportion.flatten()

    column_runs = []
    for i in range(n_runs):
        for j in range(n_gens):
            for k in range(n_language_classes):
                column_runs.append(i)
    column_runs = np.array(column_runs)

    column_generation = []
    for i in range(n_runs):
        for j in range(n_gens):
            for k in range(n_language_classes):
                column_generation.append(j)
    column_generation = np.array(column_generation)

    column_type = []
    for i in range(n_runs):
        for j in range(n_gens):
            if n_language_classes == 4:
                column_type.append('degenerate')
                column_type.append('holistic')
                column_type.append('other')
                column_type.append('compositional')
            elif n_language_classes == 5:
                column_type.append('degenerate')
                column_type.append('holistic')
                column_type.append('hybrid')
                column_type.append('compositional')
                column_type.append('other')

    data = {'run': column_runs,
            'generation': column_generation,
            'proportion': column_proportion,
            'class': column_type}

    lang_class_prop_over_gen_df = pd.DataFrame(data)

    return lang_class_prop_over_gen_df


def dataframe_to_language_stats(dataframe, n_runs, n_gens, n_language_classes):
    """
    Takes a pandas dataframe of results and turns it back into a simple results array, which only contains the
    populations' posterior probability distributions over generations.

    :param dataframe: a pandas dataframe which contains at least a column named "proportions", which contains the
    proportions of the different language classes over generations over runs.
    :param n_runs: number of runs (int)
    :param n_gens: number of generations (int)
    :param n_language_classes: the number of language classes that are distinguished (int); corresponds to global
    variable 'n_lang_classes'. This should be 4 if the old code was used (from before 13 September 2019, 1:30 pm), which
    did not yet distinguish between 'holistic' and 'hybrid' languages, and 5 if the new code was used which does make
    this distinction
    :return: a numpy array containing the proportions of the different language classes for each generation for each run
    """
    proportion_column = np.array(dataframe['proportion'])
    proportion_column_as_results = proportion_column.reshape((n_runs, n_gens, n_language_classes))
    return proportion_column_as_results


# AND NOW SOME FUNCTIONS THAT CONVERT VARIABLES TO FORMATS THAT ARE MORE SUITABLE FOR USING IN A FILENAME:

def convert_float_value_to_string(float_value):
    """
    :param float_value: float
    :return: The float converted into a string where dots are removed.
    """
    if float_value % 1. == 0:
        float_value = int(float_value)
    float_string = str(float_value)
    float_string = float_string.replace(".", "")
    return float_string


def convert_array_to_string(array):
    """
    :param array: A 1D numpy array
    :return: The numpy array converted into a string where spaces are replaced by underscores and brackets and dots are
    removed.
    """
    array_string = np.array2string(array, separator=',')
    array_string = array_string.replace(" ", "")
    array_string = array_string.replace(".", "")
    return array_string


# AND NOW FOR THE PLOTTING FUNCTIONS:

def plot_timecourse(lang_class_prop_over_gen_df, title, file_path, file_name, n_language_classes):
    """
    Takes a pandas dataframe which contains the proportions of language classes over generations and plots timecourses

    :param lang_class_prop_over_gen_df: a pandas dataframe containing four columns: 'run', 'generation', 'proportion'
    and 'class'
    :param title: The title of the condition that should be on the plot (string)
    :param file_path: path to folder in which the figure file should be saved
    :param file_name: The file name that the plot should be saved under
    :param n_language_classes: the number of language classes that are distinguished (int). This should be 4 if the old code
    was used (from before 13 September 2019, 1:30 pm), which did not yet distinguish between 'holistic' and 'hybrid'
    languages, and 5 if the new code was used which does make this distinction.
    :return: Nothing. Just saves the plot and then shows it.
    """
    sns.set_style("whitegrid")
    sns.set_context("talk")

    fig, ax = plt.subplots()

    if n_language_classes == 4:
        palette = sns.color_palette(["black", "red", "grey", "green"])
    elif n_language_classes == 5:
        palette = sns.color_palette(["black", "red", "magenta", "green", "grey"])

    sns.lineplot(x="generation", y="proportion", hue="class", data=lang_class_prop_over_gen_df, palette=palette)
    # sns.lineplot(x="generation", y="proportion", hue="class", data=lang_class_prop_over_gen_df, palette=palette, ci=95, err_style="bars")

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.ylim(-0.05, 1.05)
    plt.title(title, fontsize=22)
    plt.xlabel('Generation', fontsize=20)
    plt.ylabel('Mean proportion', fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.tight_layout()
    plt.savefig(file_path + "Timecourse_plot_" + file_name + ".pdf")
    plt.show()


def plot_barplot(lang_class_prop_over_gen_df, title, file_path, file_name, n_runs, n_gens, gen_start, n_language_classes, lang_class_baselines):
    """
    Takes a pandas dataframe which contains the proportions of language classes over generations and generates a barplot
    (excluding the burn-in period)

    :param lang_class_prop_over_gen_df: a pandas dataframe containing four columns: 'run', 'generation', 'proportion'
    and 'class'
    :param title: The title of the condition that should be on the plot (string)
    :param file_path: path to folder in which the figure file should be saved
    :param file_name: The file name that the plot should be saved under
    :param n_runs: the number of runs (int); corresponds to global variable 'runs'
    :param n_gens: the number of generations (int); corresponds to global variable 'generations'
    :param gen_start: the burn-in period that is excluded when calculating the means and confidence intervals
    :param n_language_classes: the number of language classes that are distinguished (int). This should be 4 if the old code
    was used (from before 13 September 2019, 1:30 pm), which did not yet distinguish between 'holistic' and 'hybrid'
    languages, and 5 if the new code was used which does make this distinction.
    :param lang_class_baselines: The baseline proportion for each language class, where the ordering depends on the code that was
    used, as described above.
    :return: Nothing. Just saves the plot and then shows it.
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")

    proportion_column_as_results = dataframe_to_language_stats(lang_class_prop_over_gen_df, n_runs, n_gens, n_language_classes)

    proportion_column_from_start_gen = proportion_column_as_results[:, gen_start:]

    proportion_column_from_start_gen = proportion_column_from_start_gen.flatten()

    runs_column_from_start_gen = []
    for i in range(n_runs):
        for j in range(gen_start, n_gens):
            for k in range(n_language_classes):
                runs_column_from_start_gen.append(i)
    runs_column_from_start_gen = np.array(runs_column_from_start_gen)

    generation_column_from_start_gen = []
    for i in range(n_runs):
        for j in range(gen_start, n_gens):
            for k in range(n_language_classes):
                generation_column_from_start_gen.append(j)
    generation_column_from_start_gen = np.array(generation_column_from_start_gen)

    class_column_from_start_gen = []
    for i in range(n_runs):
        for j in range(gen_start, n_gens):
            if n_language_classes == 4:
                class_column_from_start_gen.append('degenerate')
                class_column_from_start_gen.append('holistic')
                class_column_from_start_gen.append('other')
                class_column_from_start_gen.append('compositional')
            elif n_language_classes == 5:
                class_column_from_start_gen.append('degen.')
                class_column_from_start_gen.append('holistic')
                class_column_from_start_gen.append('hybrid')
                class_column_from_start_gen.append('comp.')
                class_column_from_start_gen.append('other')


    new_data_dict = {'run': runs_column_from_start_gen,
            'generation': generation_column_from_start_gen,
            'proportion': proportion_column_from_start_gen,
            'class': class_column_from_start_gen}

    lang_class_prop_over_gen_df_from_starting_gen = pd.DataFrame(new_data_dict)

    if n_language_classes == 4:
        color_palette = sns.color_palette(["black", "red", "grey", "green"])
    elif n_language_classes == 5:
        color_palette = sns.color_palette(["black", "red", "magenta", "green", "grey"])

    sns.barplot(x="class", y="proportion", data=lang_class_prop_over_gen_df_from_starting_gen, palette=color_palette)

    if n_language_classes == 4:
        plt.axhline(y=lang_class_baselines[0], xmin=0.0, xmax=0.25, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines[1], xmin=0.25, xmax=0.5, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines[2], xmin=0.5, xmax=0.75, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines[3], xmin=0.75, xmax=1.0, color='k', linestyle='--', linewidth=2)
    elif n_language_classes == 5:
        plt.axhline(y=lang_class_baselines[0], xmin=0.0, xmax=0.2, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines[1], xmin=0.2, xmax=0.4, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines[2], xmin=0.4, xmax=0.6, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines[3], xmin=0.6, xmax=0.8, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines[4], xmin=0.8, xmax=1.0, color='k', linestyle='--', linewidth=2)

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.ylim(-0.05, 1.05)
    plt.title(title, fontsize=22)
    # plt.xlabel('Language class')
    plt.xlabel('', fontsize=20)
    plt.ylabel('Mean proportion', fontsize=20)
    plt.tight_layout()

    plt.savefig(file_path + "Barplot_" + file_name + "_burn_in_" + str(gen_start) + ".pdf")
    plt.show()


###################################################################################################################
if __name__ == '__main__':

    ###################################################################################################################
    # FIRST LET'S CHECK MY LANGUAGES AND THE CLASSIFICATION OF THEM AGAINST THE SIMLANG CODE, JUST AS A SANITY CHECK:

    hypothesis_space = create_all_possible_languages(meanings, forms_without_noise)
    print("number of possible languages is:")
    print(len(hypothesis_space))

    # Let's check whether the functions in this cell work correctly by comparing the number of languages of each type we
    # get with the SimLang lab 21:

    # types_simlang = np.array(types_simlang)
    # no_of_each_type = np.bincount(types_simlang)
    # print('')
    # print("no_of_each_type ACCORDING TO SIMLANG CODE, where 0 = degenerate, 1 = holistic, 2 = other, 3 = compositional is:")
    # print(no_of_each_type)

    class_per_lang = classify_all_languages(hypothesis_space, forms_without_noise, meanings)
    print('')
    print('')
    # print("class_per_lang is:")
    # print(class_per_lang)
    no_of_each_class = np.bincount(class_per_lang.astype(int))
    print('')
    print("no_of_each_class ACCORDING TO MY CODE, where 0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, "
          "4 = other is:")
    print(no_of_each_class)

    # Hmmm, that gives us slightly different numbers! Is that caused by a problem in my
    # create_all_languages() function, or in my classify_lang() function?
    # To find out, let's compare my list of all languages to that from SimLang lab 21:

    # First, we need to change the way we represent the list of all languages to match
    # that of lab 21:

    all_langs_as_in_simlang = transform_all_languages_to_simlang_format(hypothesis_space, meanings)
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

    checks_per_language, new_log_prior = check_all_lang_lists_against_each_other(all_langs_as_in_simlang, languages_simlang, priors_simlang)
    print('')
    print('')
    # print("checks_per_language is:")
    # print(checks_per_language)
    print("np.sum(checks_per_language) is:")
    print(np.sum(checks_per_language))

    print('')
    print('')
    # print("new_log_prior is:")
    # print(new_log_prior)
    # print("np.exp(new_log_prior) is:")
    # print(np.exp(new_log_prior))
    print("new_log_prior.shape is:")
    print(new_log_prior.shape)
    print("np.exp(scipy.special.logsumexp(new_log_prior)) is:")
    print(np.exp(scipy.special.logsumexp(new_log_prior)))

    # Ok, this shows that for each language in the list of all_possible_languages generated by my own code, there is a
    # corresponding languages in the code from SimLang lab 21, so instead there must be something wrong with the way I
    # categorise the languages. Firstly, it looks like my classify_language() function underestimates the number of
    # compositional languages. So let's first have a look at which languages it classifies as compositional:
    #
    # compositional_langs_indices_my_code = np.where(class_per_lang==3)[0]
    # print('')
    # print('')
    # print("compositional_langs_indices_my_code MY CODE are:")
    # print(compositional_langs_indices_my_code)
    # print("len(compositional_langs_indices_my_code) MY CODE are:")
    # print(len(compositional_langs_indices_my_code))
    #
    # for index in compositional_langs_indices_my_code:
    #     print('')
    #     print("index MY CODE is:")
    #     print(index)
    #     print("all_possible_languages[index] MY CODE is:")
    #     print(all_possible_languages[index])
    #
    # # And now let's do the same for the languages from SimLang Lab 21:
    #
    # compositional_langs_indices_simlang = np.where(np.array(types_simlang)==3)[0]
    # print('')
    # print('')
    # print("compositional_langs_indices_simlang SIMLANG CODE are:")
    # print(compositional_langs_indices_simlang)
    # print("len(compositional_langs_indices_simlang) SIMLANG CODE are:")
    # print(len(compositional_langs_indices_simlang))
    #
    # for index in compositional_langs_indices_simlang:
    #     print('')
    #     print("index SIMLANG CODE is:")
    #     print(index)
    #     print("languages_simlang[index] SIMLANG CODE is:")
    #     print(languages_simlang[index])

    # Hmm, so it looks like instead of there being a bug in my code, there might actually be a bug in the SimLang lab 21
    # code (or rather, in the code that generated the list of types that was copied into SimLang lab 21)
    # Let's check whether maybe the holistic languages that are miscategorised as compositional in the SimLang code happen
    # to be the ones we identified as "hybrids" (i.e. kind of in between holistic and compositional) above:

    # hybrid_langs_indices_my_code = np.where(class_per_lang==2)[0]
    # print('')
    # print('')
    # print("hybrid_langs_indices_my_code MY CODE are:")
    # print(hybrid_langs_indices_my_code)
    # print("len(hybrid_langs_indices_my_code) MY CODE are:")
    # print(len(hybrid_langs_indices_my_code))
    #
    # for index in hybrid_langs_indices_my_code:
    #     print('')
    #     print("index MY CODE is:")
    #     print(index)
    #     print("all_possible_languages[index] MY CODE is:")
    #     print(all_possible_languages[index])


    ###################################################################################################################
    # NOW LET'S RUN THE ACTUAL SIMULATION:

    t0 = time.process_time()

    hypothesis_space = create_all_possible_languages(meanings, forms_without_noise)
    print("number of possible languages is:")
    print(len(hypothesis_space))


    if compressibility_bias:
        priors = new_log_prior
    else:
        priors = np.ones(len(hypothesis_space))
        priors = np.divide(priors, np.sum(priors))
        priors = np.log(priors)

    initial_dataset = create_initial_dataset(initial_language_type, b, hypothesis_space, class_per_lang, meanings)  # the data that the first generation learns from

    language_stats_over_gens_per_run = []
    data_over_gens_per_run = []
    final_pop_per_run = []
    for i in range(runs):
        print('')
        print('run '+str(i))
        language_stats_over_gens, data_over_gens, final_pop = simulation(generations, rounds, b, popsize, hypothesis_space, priors, initial_dataset, interaction, production, gamma, noise, noise_prob, all_forms_including_noisy_variants)
        language_stats_over_gens_per_run.append(language_stats_over_gens)
        data_over_gens_per_run.append(data_over_gens)
        final_pop_per_run.append(final_pop)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    pickle_file_name = "Pickle_r_" + str(runs) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_popsize_" + str(popsize) + "_mutual_u_"+str(mutual_understanding)+ "_gamma_" + str(gamma) +"_minimal_e_"+str(minimal_effort)+ "_c_"+convert_array_to_string(cost_vector)+ "_turnover_" + str(turnover) + "_bias_" +str(compressibility_bias) + "_init_" + initial_language_type + "_noise_" + str(noise) + "_" + convert_float_value_to_string(noise_prob)+"_observed_m_"+observed_meaning+"_n_l_classes_"+str(n_lang_classes)+"_CS_"+str(communicative_success_pressure)+"_"+convert_float_value_to_string(np.around(communicative_success_pressure_strength, decimals=2))+"_"+timestr
    pickle.dump(language_stats_over_gens_per_run, open(pickle_file_path + pickle_file_name + "_language_stats" + ".p", "wb"))
    pickle.dump(data_over_gens_per_run, open(pickle_file_path+pickle_file_name+"_data"+".p", "wb"))
    pickle.dump(final_pop_per_run, open(pickle_file_path + pickle_file_name + "_final_pop" + ".p", "wb"))

    lang_class_prop_over_gen_dataframe = language_stats_to_dataframe(language_stats_over_gens_per_run, runs, generations, n_lang_classes)
    lang_class_prop_over_gen_dataframe.to_pickle(pickle_file_path + pickle_file_name + "_language_stats_pandas_df" + ".pkl")
    # to unpickle this data file, run: lang_class_prop_over_gen_dataframe = pd.read_pickle(pickle_file_name+".pkl")

    fig_file_name = "r_" + str(runs) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_popsize_" + str(popsize) + "_mutual_u_"+str(mutual_understanding)+  "_gamma_" + str(gamma) +"_minimal_e_"+str(minimal_effort)+ "_c_"+convert_array_to_string(cost_vector)+ "_turnover_" + str(turnover) + "_bias_" +str(compressibility_bias) + "_init_" + initial_language_type + "_noise_" + str(noise) + "_noise_prob_" + convert_float_value_to_string(noise_prob)+"_observed_m_"+observed_meaning+"_n_l_classes_"+str(n_lang_classes)+"_CS_"+str(communicative_success_pressure)+"_"+convert_float_value_to_string(np.around(communicative_success_pressure_strength, decimals=2))

    if mutual_understanding is False and minimal_effort is False:
        if gamma == 0 and turnover is True:
            plot_title = "Learnability only"
        elif gamma > 0 and turnover is False:
            plot_title = "Expressivity only"
        elif gamma > 0 and turnover is True:
            plot_title = "Learnability and expressivity"
        if noise:
            plot_title = plot_title+" Plus Noise"
    else:
        if mutual_understanding is True and minimal_effort is False:
            plot_title = "Mutual Understanding Only"
        elif mutual_understanding is False and minimal_effort is True:
            plot_title = "Minimal Effort Only"
        elif mutual_understanding is True and minimal_effort is True:
            plot_title = "Mutual Understanding and Minimal Effort"

    plot_timecourse(lang_class_prop_over_gen_dataframe, plot_title, fig_file_path, fig_file_name, n_lang_classes)

    baseline_proportions = np.divide(no_of_each_class, len(hypothesis_space))
    print('')
    print('')
    print("baseline_proportions are:")
    print(baseline_proportions)

    plot_barplot(lang_class_prop_over_gen_dataframe, plot_title, fig_file_path, fig_file_name, runs, generations, burn_in, n_lang_classes, baseline_proportions)

    t1 = time.process_time()

    print('')
    print('')
    print("number of minutes it took to run simulation:")
    print((t1-t0)/60.)
