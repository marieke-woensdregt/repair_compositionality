import itertools
import numpy as np
import random
import scipy.special
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


# FROM SIMLANG LAB 21:
languages = [[('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')]]
types = [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]
priors = [-0.9178860550328204, -10.749415928290118, -10.749415928290118, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -12.460704095246543, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -12.460704095246543, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075, -10.749415928290118, -2.304180416152711, -11.272664072079987, -10.749415928290118, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061, -10.749415928290118, -11.272664072079987, -2.304180416152711, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075, -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -12.460704095246543, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075, -12.460704095246543, -16.95425710594061, -16.95425710594061, -16.95425710594061, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987, -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118, -11.272664072079987, -10.749415928290118, -10.749415928290118, -0.9178860550328204]


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
    0 = degenerate, 1 = holistic, 2 = other, 3 = compositional (here I'm following the
    numbering used in SimLang lab 21)
    :rtype: int
    """
    # TODO: See if I can modify this function so that it can deal with any number of forms_without_noisy_variants and
    #  meanings.
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

    # lang is compositional if it makes use of all possible forms_without_noisy_variants, *and* each form element maps
    # to the same meaning element for each form:
    elif forms[0] in lang and forms[1] in lang and forms[2] in lang and forms[
        3] in lang and lang[0][0] == lang[1][0] and lang[2][0] == lang[3][0] and lang[0][
        1] == lang[2][1] and lang[1][1] == lang[3][1]:
        return class_compositional

    # lang is holistic if it is *not* compositional, but *does* make use of all possible forms_without_noisy_variants:
    elif forms[0] in lang and forms[1] in lang and forms[2] in lang and forms[3] in lang:
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



# # Let's test our classify_language() function using some example languages from the Kirby et al. (2015) paper:
# degenerate_lang = ('aa', 'aa', 'aa', 'aa')
# print('')
# print("degenerate_lang is:")
# print(degenerate_lang)
# class_degenerate_lang = classify_language(degenerate_lang, forms_without_noise, meanings)
# print("class_degenerate_lang is:")
# print(class_degenerate_lang)
#
# holistic_lang = ('aa', 'ab', 'bb', 'ba')
# print('')
# print("holistic_lang is:")
# print(holistic_lang)
# class_holistic_lang = classify_language(holistic_lang, forms_without_noise, meanings)
# print("class_holistic_lang is:")
# print(class_holistic_lang)
#
# other_lang = ('aa', 'aa', 'aa', 'ab')
# print('')
# print("other_lang is:")
# print(other_lang)
# class_other_lang = classify_language(other_lang, forms_without_noise, meanings)
# print("class_other_lang is:")
# print(class_other_lang)
#
# compositional_lang = ('aa', 'ab', 'ba', 'bb')
# print('')
# print("compositional_lang is:")
# print(compositional_lang)
# class_compositional_lang = classify_language(compositional_lang, forms_without_noise,
#                                              meanings)
# print("class_compositional_lang is:")
# print(class_compositional_lang)


def classify_all_languages(language_list):
    """
    Classify all languages as either 'degenerate' (0), 'holistic' (1), 'other' (2) or 'compositional' (3)
    (Kirby et al., 2015)

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
    for l in range(len(language_list)):
        lang_as_in_simlang = [(meanings[x], language_list[l][x]) for x in range(len(meanings))]
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
print("np.exp(scipy.special.logsumexp(new_log_prior)) is:")
print(np.exp(scipy.special.logsumexp(new_log_prior)))
priors = new_log_prior

# Ok, this shows that for each language in the list of all_possible_languages generated by my own code, there is a
# corresponding languages in the code from SimLang lab 21, so instead there must be something wrong with the way I
# categorise the languages. Firstly, it looks like my classify_language() function underestimates the number of
# compositional languages. So let's first have a look at which languages it classifies as compositional:


# compositional_langs_indices_my_code = np.where(class_per_lang==3)[0]
# print('')
# print('')
# print("compositional_langs_indices_my_code MY CODE are:")
# print(compositional_langs_indices_my_code)
# print("len(compositional_langs_indices_my_code) MY CODE are:")
# print(len(compositional_langs_indices_my_code))
#
#
# for index in compositional_langs_indices_my_code:
#     print('')
#     print("index MY CODE is:")
#     print(index)
#     print("all_possible_languages[index] MY CODE is:")
#     print(all_possible_languages[index])
#
#
# # And now let's do the same for the languages from SimLang Lab 21:
#
# compositional_langs_indices_simlang = np.where(np.array(types)==3)[0]
# print('')
# print('')
# print("compositional_langs_indices_simlang SIMLANG CODE are:")
# print(compositional_langs_indices_simlang)
# print("len(compositional_langs_indices_simlang) SIMLANG CODE are:")
# print(len(compositional_langs_indices_simlang))
#
#
# for index in compositional_langs_indices_simlang:
#     print('')
#     print("index SIMLANG CODE is:")
#     print(index)
#     print("languages[index] SIMLANG CODE is:")
#     print(languages[index])


# Hmm, so it looks like instead of there being a bug in my code, there might actually be a bug in the SimLang lab 21
# code (or rather, in the code that generated the list of types that was copied into SimLang lab 21)








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
    creates for instance 'a_' and '_b', but not '__'.

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
    prop_to_prob_error_form_noisy = error / (len(forms_without_noise) - 1) * (1 - noise_prob) * (noise_prob / len(noisy_forms))
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

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param gamma: parameter that determines the strength of the penalty on ambiguity
    :param error: the probability of making an error in production
    :return: an utterance. That is, a single form chosen from either the global variable "forms_without_noise" (if
    noise is False) or the global variable "all_forms_including_noisy_variants" (if noise is True).
        """
    if noise:
        prop_to_prob_per_form_array = production_likelihoods_with_noise(language, topic, gamma, error, noise_prob)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(all_forms_including_noisy_variants, p=prob_per_form_array)
    else:
        prop_to_prob_per_form_array = production_likelihoods_kirby_et_al(language, topic, gamma, error)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(forms_without_noise, p=prob_per_form_array)
    return utterance




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
    interpretation = random.choice(possible_interpretations)
    return interpretation




def update_posterior(log_posterior, topic, utterance):
    """
    Takes a LOG posterior probability distribution and a <topic, utterance> pair, and updates the posterior probability distribution accordingly

    :param log_posterior: 1D numpy array containing LOG posterior probability values for each hypothesis
    :param topic: a topic (string from the global variable meanings)
    :param utterance: an utterance (string from the global variable forms (can be a noisy form if parameter noise is True)
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

    new_log_posterior_normalized = np.subtract(new_log_posterior, scipy.special.logsumexp(new_log_posterior))

    return new_log_posterior_normalized


# print('')
# print('')
# example_prior = np.array([1./len(all_possible_languages) for x in range(len(all_possible_languages))])
# example_log_prior = np.log(example_prior)
# print("np.exp(example_log_prior) is:")
# print(np.exp(example_log_prior))
# print("np.exp(scipy.special.logsumexp(example_log_prior)) is:")
# print(np.exp(scipy.special.logsumexp(example_log_prior)))
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
# print("np.exp(scipy.special.logsumexp(new_log_posterior_normalized)) is:")
# print(np.exp(scipy.special.logsumexp(new_log_posterior_normalized)))



def new_population(popsize):
    """
    Creates a new population of agents, where each agent simply consists of the prior probability distribution (which is assumed to be defined as a global variable called 'priors')

    :param popsize: the number of agents desired in the new population
    :return: 2D numpy array, with agents on the rows, and hypotheses (or rather their corresponding prior probabilities)
    on the columns.
    """
    population = [priors for x in range(popsize)]
    population = np.array(population)
    return population





def log_roulette_wheel(normedlogs):
    """
    Samples an index from a list of LOG probabilities, where each index has a probability proportional to their probability of being chosen

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
    return all_possible_languages[log_roulette_wheel(log_posterior)]


def population_communication(population, rounds):
    """
    Takes a population, makes it communicate for a number of rounds (where agents' posterior probability distribution
    is updated every time the agent gets assigned the role of hearer)

    :param population: a population (1D numpy array), where each agent is simply a LOG posterior probability
    distribution
    :param rounds: the number of rounds for which the population should communicate
    :return: the data that was produced during the communication rounds, as a list of (meaning, signal) tuples
    """
    # random_parent_index = np.random.choice(np.arange(len(population)))
    data = []
    for i in range(rounds):
        # if len(population) == 2:
        #     if i % 2 == 0:
        #         speaker_index = 0
        #         hearer_index = 1
        #     else:
        #         speaker_index = 1
        #         hearer_index = 0
        # else:
        pair_indices = np.random.choice(np.arange(len(population)), size=2, replace=False)
        speaker_index = pair_indices[0]
        hearer_index = pair_indices[1]
        meaning = random.choice(meanings)
        signal = produce(sample(population[speaker_index]), meaning, gamma, error)  # whenever a speaker is called upon
        # to produce a signal, they first sample a language from their posterior probability distribution. So each agent
        # keeps updating their language according to the data they receive from their communication partner.
        population[hearer_index] = update_posterior(population[hearer_index], meaning, signal)  # (Thus, in this
        # simplified version of the model, agents are still able to "track changes in their partners' linguistic
        # behaviour over time

        # if speaker_index == random_parent_index:
        data.append((meaning, signal))
    return data



def dataset_from_language(language):
    """
    Takes a language and generates a balanced minimal dataset from it, in which each possible meaning occurs exactly once, combined with its corresponding form.

    :param language: a language (list of forms_without_noisy_variants that has same length as the global variable
    meanings, where each form is mapped to the meaning at the corresponding index)
    :return: a dataset (list containing tuples, where each tuple is a meaning-form pair, with the meaning followed by the form)
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

    :param desired_class: 'degenerate', 'holistic', 'other', or 'compositional'
    :return: a dataset (list containing tuples, where each tuple is a meaning-form pair, with the meaning followed by the form) from a randomly chosen language of the desired class
    """
    if desired_class == 'degenerate':
        class_index = 0
    elif desired_class == 'holistic':
        class_index = 1
    elif desired_class == 'other':
        class_index = 2
    elif desired_class == 'compositional':
        class_index = 3
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

    :param population: a population (1D numpy array), where each agent is simply a LOG posterior probability distribution
    :return: a list containing the overall average posterior probability assigned to each class of language in the population, where index 0 = degenerate, index 1 = holistic, index 2 = other, and index 3 = compositional
    (this ordering has to correspond to that defined in the classify_language() function!)
    """
    # stats = [0., 0., 0., 0.]  # degenerate, holistic, other, compositional
    stats = np.zeros(4)  # degenerate, holistic, other, compositional
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
    Runs the full simulation and returns the total amount of posterior probability that is assigned to each language class over generations (results) as well as the data that each generation produced (data)

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
                # if bottleneck != len(data):
                #     raise ValueError(
                #         "UH-OH! data should have the same size as the bottleneck b")
                # meaning, signal = data[k]
                meaning, signal = random.choice(data)
                population[j] = update_posterior(population[j], meaning, signal)
        data = population_communication(population, rounds)
        results.append(language_stats(population))
        if turnover:
            population = new_population(popsize)
    return results, data



def results_to_dataframe(results, runs, generations):
    """
    Takes a results list and puts it in a pandas dataframe together with other relevant variables (runs, generations,
    and language class)

    :param results: a list containing proportions for each of the 4 language classes, for each generation, for each run
    :return: a pandas dataframe containing four columns: 'run', 'generation', 'proportion' and 'class'
    """
    column_proportion = np.array(results)
    column_proportion = column_proportion.flatten()

    column_runs = []
    for i in range(runs):
        for j in range(generations):
            for k in range(4):
                column_runs.append(i)
    column_runs = np.array(column_runs)

    column_generation = []
    for i in range(runs):
        for j in range(generations):
            for k in range(4):
                column_generation.append(j)
    column_generation = np.array(column_generation)

    column_type = []
    for i in range(runs):
        for j in range(generations):
            column_type.append('degenerate')
            column_type.append('holistic')
            column_type.append('other')
            column_type.append('compositional')

    data = {'run': column_runs,
            'generation': column_generation,
            'proportion': column_proportion,
            'class': column_type,
            }

    lang_class_prop_over_gen_df = pd.DataFrame(data)

    return lang_class_prop_over_gen_df





def plot_graph(lang_class_prop_over_gen_df, plot_title, fig_file_title):
    """
    Takes a list of language stats over generations (results) and plots a timecourse graph

    :param results: A list of language stats over generations, containing n_runs sublists which each contain n_generations, which each contain 4 numbers (where index 0 = degenerate, index 1 = holistic, index 2 = other, and index 3 = compositional)
    :param plot_title: The title of the condition that should be on the plot (string)
    :param fig_file_title: The file name that the plot should be saved under
    :return: Nothing. Just saves the plot and then shows it.
    """
    palette = sns.color_palette(["black", "red", "grey", "green"])

    sns.lineplot(x="generation", y="proportion", hue="class", data=lang_class_prop_over_gen_df, palette=palette)
    # sns.lineplot(x="generation", y="proportion", hue="class", data=lang_class_prop_over_gen_df, palette=palette, ci=95, err_style="bars")

    plt.ylim(-0.05, 1.05)
    plt.title(plot_title)
    plt.xlabel('Generation')
    plt.ylabel('Mean proportion')
    plt.legend()
    plt.grid()
    plt.savefig(fig_file_title + ".pdf")
    plt.show()



gamma = 2  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for "Learnability Only" condition, and gamma = 2 for both "Expressivity Only" and "Learnability and Expressivity" conditions
turnover = True  # determines whether new individuals enter the population or not
b = 20  # the bottleneck (i.e. number of meaning-form pairs the each pair gets to see during training (Kirby et al. used a bottleneck of 20 in the body of the paper.
rounds = 2*b  # Kirby et al. (2015) used rounds = 2*b, but SimLang lab 21 uses 1*b
popsize = 2  # If I understand it correctly, Kirby et al. (2015) used a population size of 2: each generation is simply a pair of agents.
runs = 10  # the number of independent simulation runs (Kirby et al., 2015 used 100)
generations = 1000  # the number of generations (Kirby et al., 2015 used 100)
initial_dataset = create_initial_dataset('holistic', b)  # the data that the first generation learns from
noise = False  # parameter that determines whether environmental noise is on or off
noise_prob = 0.1  # the probability of environmental noise masking part of an utterance
# proportion_measure = 'posterior'  # the way in which the proportion of language classes present in the population is
# measured. Can be set to either 'posterior' (where we directly measure the total amount of posterior probability
# assigned to each language class), or 'sampled' (where at each generation we make all agents in the population pick a
# language and we count the resulting proportions.



if __name__ == '__main__':

    t0 = time.clock()

    results = []
    for i in range(runs):
        print('')
        print('run '+str(i))
        results.append(simulation(generations, rounds, b, popsize, initial_dataset)[0])


    lang_class_prop_over_gen_df = results_to_dataframe(results)
    print('')
    print('')
    print("lang_class_prop_over_gen_df is:")
    print(lang_class_prop_over_gen_df)


    pickle_file_title = "Pickle_n_runs_" + str(runs) +"_n_gens_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_gamma_" + str(gamma) + "_turnover_" + str(turnover) + "_noise_" + str(noise) + "_noise_prob_" + str(noise_prob)#+"_prop_measure_"+proportion_measure
    lang_class_prop_over_gen_df.to_pickle(pickle_file_title+".pkl")

    # to unpickle the data after it's been saved, simply run: lang_class_prop_over_gen_df = pd.read_pickle(pickle_file_title+".pkl")


    fig_file_title = "Plot_n_runs_" + str(runs) +"_n_gens_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_gamma_" + str(gamma) + "_turnover_" + str(turnover) + "_noise_" + str(noise) + "_noise_prob_" + str(noise_prob)#+"_prop_measure_"+proportion_measure
    if gamma == 0 and turnover == True:
        plot_title = "Learnability only"
    elif gamma > 0 and turnover == False:
        plot_title = "Expressivity only"
    elif gamma > 0 and turnover == True:
        plot_title = "Learnability and expressivity"
    if noise:
        plot_title = plot_title+" Plus Noise"
    plot_graph(lang_class_prop_over_gen_df, plot_title, fig_file_title)


    t1 = time.clock()

    print('')
    print('')
    print("number of minutes it took to run simulation:")
    print((t1-t0)/60.)
