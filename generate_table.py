#!/usr/bin/env python
#load_ext line_profiler

import sys 
import numpy as np
from matplotlib import pyplot as plt

def gen_line(dataset, depth, algostr):
     L=[]
     file_path = 'forests/rootedFrequentTrees/'+dataset+'/Results_NoLeafEdgesWithSplitValues/leq6/RF_'+depth+'_accuracy.csv'
     with open(file_path, "r") as f:
          file = f.readlines()
     for i in range(1,2):
          parts = file[i].split(',')
          L.append(float(parts[4]))


     file_path= 'SizeComparison/nodesCount_'+dataset+'_RandomForestClassifier.csv'
     with open(file_path, "r") as f:
          file = f.readlines()
     linenum=int(depth)//5
     L+=[int(file[linenum].split(',')[1])]

     file_path= 'InferenceComparison/'+dataset+'/'+dataset+'_comparisons_count.csv'
     with open(file_path, "r") as f:
          file = f.readlines()
     linenum=int(depth)//5-1 +480
     L+=[file[linenum].split(',')[1]]

     file_path = 'LearningAlgoComparisonPruning/best_accuracy_'+dataset+'_comparison_pruning.csv'
     with open(file_path, "r") as f:
          file = f.readlines()
     for pruning in [True, False]:
          linenum = 2
          if not pruning:
               linenum += 32
          if algostr == "_LogReg":
               linenum += 10
          if algostr == "_LinearSVM":
               linenum += 20
          linenum+=int(depth)//5 +1
          if pruning:
               infop = file[linenum].split(',')
          else:
               info = file[linenum].split(',')

     acc = float(info[1])
     accp = float(infop[1])

     size = int(info[2])
     sizep = int(infop[2])

     infer = (info[3])
     inferp = (infop[3])

     t = int(info[0].split('t')[1])
     tp = int(infop[0].split('_t')[1])

     sigma = (infop[0].split('_t')[0].split('_sigma_')[1].replace('_', '.'))

     L+=[t, acc, size, infer, sigma, tp, accp, sizep, inferp]

     string = dataset
     for number in L:
          if isinstance(number, float):
               adds = "{:.3f}".format(number)
          else:
               adds = str(number)
          string += " & " + adds
     string += " \\\\"
     return string.replace("satlog","statlog")

def gen_table(algostr):
     for depth in ["5","10", "15", "20"]:
          print('\hline')
          print('\multicolumn{13}{c}{Max depth = '+depth+'} \\\\')
          print('\hline')
          for dataset in ["adult", "bank", "mnist", "spambase", "satlog"]:
               line = gen_line(dataset, depth, algostr)
               print(line)

gen_table("_LinearSVM")
