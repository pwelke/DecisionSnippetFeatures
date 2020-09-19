#!/usr/bin/env python
#load_ext line_profiler

import sys 
import numpy as np
from matplotlib import pyplot as plt

dataset = sys.argv[1]
if len(sys.argv)<3:
     sigma = ""
else:
     sigma = sys.argv[2]

file_path = 'SizeComparison/nodesCount_accuracy_'+dataset+'_NoLeafEdgesWithSplitValues_leq6.csv'

with open(file_path, "r") as f:
     file = f.readlines()
'''        
sizes = []
accuracies = []
skipped_header = False

for line in file:
    parts = line.split(',')
    if skipped_header:
        sizes.append(int(parts[1]))
        accuracies.append(float(parts[2]))
    else:
        skipped_header = True
'''

sizes_RF = []; sizes_0 = []; sizes_1 = []; sizes_2 = []; sizes_3 = []
acc_RF = []; acc_0 = []; acc_1 = []; acc_2 = []; acc_3 = []

for i in range(1,len(file)-4):
    parts = file[i].split(',')
    if i % 5 == 1: 
        sizes_RF.append(int(parts[1]))
        acc_RF.append(float(parts[2]))
    if i % 5 == 2: 
        sizes_0.append(int(parts[1]))
        acc_0.append(float(parts[2]))
    if i % 5 == 3: 
        sizes_1.append(int(parts[1]))
        acc_1.append(float(parts[2]))
    if i % 5 == 4: 
        sizes_2.append(int(parts[1]))
        acc_2.append(float(parts[2]))
    if i % 5 == 0: 
        sizes_3.append(int(parts[1]))
        acc_3.append(float(parts[2]))
       
'''
#remove the last 4 elements
sizes = sizes[:-4]
accuracies = accuracies[:-4]
'''

plt.plot(sizes_RF, acc_RF, '.', label='RF', alpha=0.5)
if sigma in ["", "0"]:
     plt.plot(sizes_0, acc_0, '.', label='sigma_0_0', alpha=0.5)
if sigma in ["", "1"]:
     plt.plot(sizes_1, acc_1, '.', label='sigma_0_1', alpha=0.5)
if sigma in ["", "2"]:
     plt.plot(sizes_2, acc_2, '.', label='sigma_0_2', alpha=0.5)
if sigma in ["", "3"]:
     plt.plot(sizes_3, acc_3, '.', label='sigma_0_3', alpha=0.5)
if sigma=="uni":
     plt.plot(sizes_0+sizes_1+sizes_2+sizes_3, acc_0+acc_1+acc_2+acc_3, '.', label="pruned", alpha=0.5)
plt.legend(loc='lower right')
plt.show()
