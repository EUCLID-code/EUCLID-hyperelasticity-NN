
import torch

torch.manual_seed(123)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

#numpy
import numpy as np
from numpy import inf
np.random.seed(123)

#matplotlib
import matplotlib.pyplot as plt

#scipy
import scipy
from scipy import sparse

#pandas
import pandas as pd

#others:
import os
import sys
from contextlib import contextmanager
import shutil
from distutils.dir_util import copy_tree
from pickle import dump, load
