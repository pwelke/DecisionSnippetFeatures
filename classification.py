#!/usr/bin/env python
#load_ext line_profiler
import sys
import csv,operator,sys,os
import numpy as np
import sklearn
import json
import FeatureGenerators.ReadData
import time
from functools import reduce
import warnings; warnings.simplefilter('ignore')  #do not show warnings in the output (like deprecation warnings)

#sys.path.append('arch-forest/data/adult/')
#sys.path.append('arch-forest/data/bank/')
#sys.path.append('arch-forest/data/wine-quality/')
sys.path.append('arch-forest/data/spambase/')
sys.path.append('arch-forest/data/satlog/')
sys.path.append('arch-forest/data/')
sys.path.append('arch-forest/code/')
import trainForest
import Tree
import FeatureGenerators.DecisionSnippetFeatures

from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataPath = "arch-forest/data/"
frequentTreesPath = "forests/rootedFrequentTrees/"
resultsPath = "forests/rootedFrequentTrees/"

def classify(dataSet, algo):

    #import train data
    from FeatureGenerators.ReadData import readDataAdult,readDataBank, readWine, readWineTest, readDataSpambase, readDataMagic,readDataCovertype, readDataMnist, readDataSatlog, readDataSensorlessDrive
    #dataSet='satlog'
    if (dataSet == 'adult'):
        X_train,Y_train = FeatureGenerators.ReadData.readDataAdult('train')
        X_test,Y_test = FeatureGenerators.ReadData.readDataAdult('test')
    if (dataSet == 'spambase'):
        X_train,Y_train = FeatureGenerators.ReadData.readDataSpambase('train')
        X_test,Y_test = FeatureGenerators.ReadData.readDataSpambase('test')        
    if (dataSet == 'letter'):
        X_train,Y_train = FeatureGenerators.ReadData.readDataLetter('train')
        X_test,Y_test = FeatureGenerators.ReadData.readDataLetter('test')  
    if (dataSet == 'bank'):
        X_train,Y_train = FeatureGenerators.ReadData.readDataBank('train')
        X_test,Y_test = FeatureGenerators.ReadData.readDataBank('test')
    if (dataSet == 'magic'):
        X_train,Y_train = FeatureGenerators.ReadData.readDataMagic('train')
        X_test,Y_test = FeatureGenerators.ReadData.readDataMagic('test')
    if (dataSet == 'covertype'):
        X_train,Y_train = FeatureGenerators.ReadData.readDataCovertype('train')
        X_test,Y_test = FeatureGenerators.ReadData.readDataCovertype('test') 
    if (dataSet == 'mnist'):
        X_train,Y_train = FeatureGenerators.ReadData.readDataMnist('train')
        X_test,Y_test = FeatureGenerators.ReadData.readDataMnist('test')    
    if (dataSet == 'satlog'):
        X_train,Y_train = FeatureGenerators.ReadData.readDataSatlog('train')
        X_test,Y_test = FeatureGenerators.ReadData.readDataSatlog('test')
    if (dataSet == 'sensorless-drive'):
        X_train,Y_train = FeatureGenerators.ReadData.readDataSensorlessDrive('train')
        X_test,Y_test = FeatureGenerators.ReadData.readDataSensorlessDrive('test')     
    if (dataSet == 'wine-quality'):
            X_train,Y_train = FeatureGenerators.ReadData.readWine()
            #X_test,Y_test = ReadData.readWineTest()
    print(len(X_test))


    # Classification
    scoring_function = 'accuracy'
    pattern_max_size=6
    variant = 'NoLeafEdgesWithSplitValues'

    start_time = time.time()
    model = GaussianNB()
    normalfeatures_nb_cv_score = cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring_function)
    normal_time = time.time() - start_time
    normal_score = normalfeatures_nb_cv_score.mean()
    print('normal: '+str(normal_score))

    for rf_depth in (5,10,15,20):

        results_list = []
        time_list = []

        start_time = time.time()
        model = DecisionTreeClassifier(max_depth=rf_depth)
        dt_cv_score = cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring_function)
        dt_time = time.time() - start_time
        dt_score = dt_cv_score.mean()
        print('DT: '+str(dt_score))

        start_time = time.time()    
        model = RandomForestClassifier(max_depth=rf_depth, n_estimators=100)
        rf_cv_score = cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring_function)
        rf_time = time.time() - start_time
        rf_score = rf_cv_score.mean()
        print('RF: '+str(rf_score))


        for frequency in range(2,26):

            for pruning in ['','_pruned_with_sigma_0_0','_pruned_with_sigma_0_1','_pruned_with_sigma_0_2','_pruned_with_sigma_0_3']:

                start_time = time.time()
                rootedFrequentTrees = "RF_"+str(rf_depth)+pruning+"_t"+str(frequency)
                f = open(frequentTreesPath+dataSet+'/'+variant+'/leq'+str(pattern_max_size)+'/'+rootedFrequentTrees+'.json')
                frequentpatterns = json.load(f)
                f.close()

                #if (frequency < 4):


                dsf = FeatureGenerators.DecisionSnippetFeatures.FrequentSubtreeFeatures(map(lambda x: x['pattern'], frequentpatterns[-100:]))
                #dsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(frequentpatterns)

                fts = dsf.fit_transform(X_train,0)
                fts_test = dsf.fit_transform(X_test,0)
                #print(fts)

                from sklearn.preprocessing import OneHotEncoder
                fts_onehot = OneHotEncoder(n_values=dsf.get_n_values()).fit_transform(fts)
                fts_onehot_test = OneHotEncoder(n_values=dsf.get_n_values()).fit_transform(fts_test)


                #model: KNN
                ''' 
                model = KNeighborsClassifier(n_neighbors=25,metric='euclidean')
                model.fit( fts_onehot.toarray(),Y_train)
                y_pred = model.predict(fts_onehot_test.toarray())


                dsf_score = model.score(fts_onehot_test.toarray(),Y_test)
                dsf_time = time.time() - start_time


                print('t'+str(frequency )+pruning+'_DSF: '+str(dsf_score))
                '''

                #model = svm.SVC(kernel='linear',C=1.0)
                if algo =='LinearSVM':
                    #'''
                    model = LinearSVC()
                    model.fit( fts_onehot.toarray(),Y_train)
                    y_pred = model.predict(fts_onehot_test.toarray())


                    dsf_score = model.score(fts_onehot_test.toarray(),Y_test)
                    dsf_time = time.time() - start_time
                    print('t'+str(frequency )+pruning+'_DSF: '+str(dsf_score))
                    #'''




                #print(fts_onehot.toarray()[0])
                #print(fts_onehot_test.toarray()[0])

                #model: Log Reg
                #'''
                if algo=='LogReg':
                    model = LogisticRegression()
                    model.fit( fts_onehot.toarray(),Y_train)
                    y_pred = model.predict(fts_onehot_test.toarray())


                    dsf_score = model.score(fts_onehot_test.toarray(),Y_test)
                    dsf_time = time.time() - start_time
                    print('t'+str(frequency )+pruning+'_DSF: '+str(dsf_score))
                    #for i in range (0,20):
                    #   print(Y_test[i])
                    #    print(y_pred[i])
                    #'''

                    #y_pred = model.predict(fts_onehot_test.toarray())
                    #conf_mat = confusion_matrix(Y_test,y_pred)
                    #print(conf_mat)
                    #print(classification_report(Y_test,y_pred))


                #model: Naive Bayes 
                if algo=='':
                    model = GaussianNB()
                    fts_onehot_nb_cv_score = cross_val_score(model, fts_onehot.toarray(), Y_train, cv=5, scoring=scoring_function)
                    dsf_time = time.time() - start_time
                    #print(fts_onehot_nb_cv_score)
                    dsf_score = fts_onehot_nb_cv_score.mean()
                    print('t'+str(frequency )+pruning+'_DSF: '+str(dsf_score))
                #'''



                result = "t"+str(frequency)+pruning+","+str(dsf_score)+","+str(normal_score)+","+str(dt_score)+","+str(rf_score)+",\n"
                results_list.append(result)
                times = "t"+str(frequency)+pruning+","+str(dsf_time)+","+str(normal_time)+","+str(dt_time)+","+str(rf_time)+",\n"
                time_list.append(times)

        if algo=='':
            algostr=''
        else:
            algostr='_'+algo
        
        #f= open(resultsPath+dataSet+'/Results_'+variant+'/leq'+str(pattern_max_size)+'/'+'RF_'+str(rf_depth)+'_'+scoring_function+'_KNN.csv',"a")
        f= open(resultsPath+dataSet+'/Results_'+variant+'/leq'+str(pattern_max_size)+'/'+'RF_'+str(rf_depth)+'_'+scoring_function+algostr+'.csv',"a")
        for line in results_list:
            f.write(line)
        f.close()

        #f= open(resultsPath+dataSet+'/Results_'+variant+'/leq'+str(pattern_max_size)+'/'+'RF_'+str(rf_depth)+'_'+scoring_function+'_time_KNN.csv',"a")
        f= open(resultsPath+dataSet+'/Results_'+variant+'/leq'+str(pattern_max_size)+'/'+'RF_'+str(rf_depth)+'_'+scoring_function+algostr+'_time.csv',"a")
        for line in time_list:
            f.write(line)
        f.close()


if len(sys.argv) < 2:
    print('arguments missing')
    exit()
if len(sys.argv) ==2:
    algo=''
else:
    algo=sys.argv[2]

ds = sys.argv[1]

classify(ds,algo)
