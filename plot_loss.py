#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:21:06 2021

@author: haoqiwang
"""

import numpy as np
from matplotlib import pyplot as plt


y = [109.747375,
101.690315,
66.426361,
33.218933,
27.365578,
23.829790,
21.210890,
19.014378,
16.894600,
15.307448,
14.066682,
12.365193,
11.444667,
10.366002,
9.349622,
9.154653,
7.909101,
7.271635,
6.658885,
6.145379]

fig = plt.figure()
plt.plot(y)
# fig.suptitle('Loss')
plt.xlabel('n_epochs')
plt.ylabel('Loss')