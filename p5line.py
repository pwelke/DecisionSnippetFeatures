#!/usr/bin/env python
dataset = "bank"
algostr="_LinearSVM"

import csv
import matplotlib.pyplot as plt
import numpy as np


variant = "NoLeafEdgesWithSplitValues"
scoring_function = 'accuracy'
pattern_max_size = 6
filesPath = "forests/rootedFrequentTrees"
filesPath_RF = "arch-forest/data"
resultsPath = "LearningAlgoComparisonPruning"




accuracy_list_naive_bayes_pruned = []
accuracy_list_log_reg_pruned = []
accuracy_list_svm_pruned = []
accuracy_list_knn_pruned = []
accuracy_list_naive_bayes_unpruned = []
accuracy_list_log_reg_unpruned = []
accuracy_list_svm_unpruned = []
accuracy_list_knn_unpruned = []
rf_list = []





List=[]
for rf_depth in (5,10,15,20):
    List=[]
    with open(filesPath+'/'+dataset+'/Results_'+variant+'/leq'+str(pattern_max_size)+'/'+'RF_'+str(rf_depth)+'_'+scoring_function+algostr+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        line_count = 0
        for row in csv_reader:
            rowStr = str(row).split(',')
            if line_count < 120:
                List.append(float(rowStr[1]))
                line_count+=1
    csv_file.close()

    table = np.reshape(np.array(List), (-1,5))
    print(table)

    for line in table:
        plt.plot([-0.2, 0, 0.1, 0.2, 0.3], line)

    plt.show()