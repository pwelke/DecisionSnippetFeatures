#!/usr/bin/env python

import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

dataset = sys.argv[1]
algostr = ''   #"_LinearSVM"
if len(sys.argv) > 2:
    algostr = sys.argv[2]



variant = "NoLeafEdgesWithSplitValues"
scoring_function = 'accuracy'
pattern_max_size = 6
filesPath = "forests/rootedFrequentTrees"
filesPath_RF = "arch-forest/data"
resultsPath = "Evaluation_best_accuracy/"




accuracy_list_naive_bayes_pruned = []
accuracy_list_log_reg_pruned = []
accuracy_list_svm_pruned = []
accuracy_list_knn_pruned = []
accuracy_list_naive_bayes_unpruned = []
accuracy_list_log_reg_unpruned = []
accuracy_list_svm_unpruned = []
accuracy_list_knn_unpruned = []
rf_list = []

for algostr in ['', '_LogReg', '_LinearSVM']:
    plot_title = dataset
    if len(algostr) > 2:
        plot_title += algostr
    else:
        plot_title += '_NaiveBayes'

    x_str = ['unpruned','sig 0.0','sig 0.1','sig 0.2', 'sig 0.3']


    List=[]
    plt.title('test')
    for rf_depth in (5,10,15,20):

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
    #print(table)
    mean_table = np.mean(table,axis=0)
    var_table = np.var(table,axis=0)
    max_table = np.max(table,axis=0)
    best_index = np.argmax(max_table)
    #print('mean',mean_table)
    #print('var',var_table)
    #print('max',max_table,'\n')

    #for line in table:
    #    plt.plot(x_str, line, alpha=0.3)
    #plt.errorbar(x_str, mean_table,yerr=var_table, capsize=3, color='r')
    plt.plot(x_str,max_table, 'o-', label=plot_title.split('_')[1])
    plt.plot([x_str[best_index]],[max_table[best_index]],marker='o',color='red')
    #plt.title()


    #plt.tight_layout()
    # Option 1
    # QT backend
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()

    # Option 2
    # TkAgg backend
    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())

    # Option 3
    # WX backend
    #manager = plt.get_current_fig_manager()
    #manager.frame.Maximize(True)
    #manager.full_screen_toggle()

plt.title(dataset)
plt.legend(loc='best')
#plt.show()
plt.savefig(resultsPath + dataset)
