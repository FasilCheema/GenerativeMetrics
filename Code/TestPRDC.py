'''
Author: Fasil Cheema 
Date  : July 26, 2021
Purpose: Test previous metrics using toy models
'''

import numpy as np
from prdc import compute_prdc

real_val = [-1,-0.5,-0.5,-0.5,0,0,0,0,0,0.5,0.5,0.5,1]
fake_val = [-0.5,0,0,0,0.5,0.5,0.5,0.5,0.5,1,1,1,1.5]

#Turn these into an array 