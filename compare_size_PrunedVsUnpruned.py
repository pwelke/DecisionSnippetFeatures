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

file_path = 'SizeComparison/nodesCount_'+dataset+'_NoLeafEdgesWithSplitValues_leq6.csv'

with open(file_path, "r") as f:
     file = f.readlines()
        
x_str = ['sig 0.0','sig 0.1','sig 0.2','sig 0.3']
depth_list = []
sizes_RF = []; sizes_0 = []; sizes_1 = []; sizes_2 = []; sizes_3 = []

for i in range(1,len(file)):
    
    parts = file[i].split(',')
    if i % 5 == 1: 
        sizes_RF.append(int(parts[1]))
    if i % 5 == 2: 
        sizes_0.append(int(parts[1]))
    if i % 5 == 3: 
        sizes_1.append(int(parts[1]))
    if i % 5 == 4: 
        sizes_2.append(int(parts[1]))
    if i % 5 == 0: 
        sizes_3.append(int(parts[1]))
       
    if i % 120 == 0:
        ratios = np.array([ [ sizes_0[j]/sizes_RF[j], sizes_1[j]/sizes_RF[j], sizes_2[j]/sizes_RF[j], sizes_3[j]/sizes_RF[j] ] for j in range(len(sizes_RF)) ])          
        mean_ratios = np.mean(ratios,axis=0)
        #var_ratios = np.var(ratios,axis=0)
        #plt.errorbar(x_str, mean_ratios,yerr=var_ratios, capsize=3, label='max_depth='+str(int(i/24)))
        plt.plot(x_str, mean_ratios, 'o-', label='max_depth='+str(int(i/24)))
            
        sizes_RF = []; sizes_0 = []; sizes_1 = []; sizes_2 = []; sizes_3 = []



plt.title('size ratios '+dataset)
plt.legend(loc='best')
plt.show()

