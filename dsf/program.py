
# %%

import csv
import operator
import sys
import os
import numpy as np
import sklearn
import json
import time
from functools import reduce
import subprocess
import pickle

import matplotlib.pyplot as plt

from sklearn.utils.estimator_checks import check_estimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import ReadData
import cString2json as cString2json
import json2graphNoLeafEdgesWithSplitValues as json2graphNoLeafEdgesWithSplitValues
from fitModels import fitModels
import DecisionSnippetFeatures as DecisionSnippetFeatures

# %% Parameters

dataPath = "./data/"
forestsPath = "./tmp/forests/"
snippetsPath = "./tmp/snippets/"
resultsPath = "./tmp/results/"


# current valid options are ['sensorless', 'satlog', 'mnist', 'magic', 'spambase', 'letter', 'bank', 'adult', 'drinking']
dataSet = 'magic'
# dataSet = 'adult'
# dataSet = 'drinking'

# possible forest_types ['RF', 'DT', 'ET']
forest_types = ['RF']
forest_depths = [5, 10, 15, 20]
forest_size = 25

maxPatternSize = 6
minThreshold = 2
maxThreshold = 25

scoring_function = 'accuracy'

# learners that are to be used on top of Decision Snippet Features
learners = {'DSF_NB': MultinomialNB,
            'DSF_SVM': LinearSVC, 
            'DSF_LR': LogisticRegression}

# specify parameters that are given at initialization
learners_parameters = {'DSF_NB': {},
                       'DSF_SVM': {'max_iter': 10000},
                       'DSF_LR': {'max_iter': 1000}}


# for quick debugging, let the whole thing run once. Afterwards, you may deactivate individual steps
# each step stores its output for the subsequent step(s) to process
run_fit_models = True
run_mining = True
run_training = True
run_eval = True

verbose = True

# %% load data

X_train, Y_train = ReadData.readData(dataSet, 'train', dataPath)
X_test, Y_test = ReadData.readData(dataSet, 'test', dataPath)


# %% create forest data, evaluate and report accuracy on test data

if run_fit_models:
    fitModels(roundSplit=True, 
              XTrain=X_train, YTrain=Y_train, 
              XTest=X_test, YTest=Y_test, 
              createTest=False, model_dir=os.path.join(forestsPath, dataSet), types=forest_types)


# %% compute decision snippets

if run_mining:

    def pattern_frequency_filter(f_patterns, frequency, f_filtered_patterns):
        pattern_count = 0
        for line in f_patterns:
            tokens = line.split('\t')
            if int(tokens[0]) >= frequency:
                f_filtered_patterns.write(line)
                pattern_count += 1
        return pattern_count


    # translate json to graph files
    for json_file in filter(lambda x: x.endswith('.json'), os.listdir(os.path.join(forestsPath, dataSet))):
        
        graph_file = json_file[:-4] + 'graph'
        with open(os.path.join(forestsPath, dataSet, json_file), 'r') as f_in:
            with open(os.path.join(forestsPath, dataSet, graph_file), 'w') as f_out:
                json2graphNoLeafEdgesWithSplitValues.main(f_in, f_out)


    # run frequent pattern mining
    if not os.path.exists(os.path.join(snippetsPath, dataSet)):
        os.makedirs(os.path.join(snippetsPath, dataSet))

    for graph_file in filter(lambda x: x.endswith('.graph'), os.listdir(os.path.join(forestsPath, dataSet))):
        
        # pattern mining for smallest minThreshold
        print(f"mining {minThreshold}-frequent patterns for {graph_file}")
        pattern_file=os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{minThreshold}.patterns')
        feature_file=os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{minThreshold}.features')
        log_file=os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{minThreshold}.logs')

        args = ['./lwgr', '-erootedTrees', '-mbfs', f'-t{minThreshold}', f'-p{maxPatternSize}', 
                f'-o{pattern_file}', os.path.join(forestsPath, dataSet, graph_file)]
        with open(feature_file, 'w') as f_out:
            with open(log_file, 'w') as f_err:        
                subprocess.run(args, stdout=f_out, stderr=f_err)

        # filtering of patterns for larger thresholds
        print(f"filtering more frequent patterns for {graph_file}")
        for threshold in range(maxThreshold, minThreshold, -1):
            filtered_pattern_file=os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{threshold}.patterns')
            pattern_count = -1
            with open(pattern_file, 'r') as f_patterns:
                with open(filtered_pattern_file, 'w') as f_filtered_patterns:
                    pattern_count = pattern_frequency_filter(f_patterns, threshold, f_filtered_patterns)
            
            # if there are no frequent patterns for given threshold, remove the file.
            if pattern_count == 0:
                os.remove(filtered_pattern_file)
            else:
                if verbose:
                    print(f'threshold {threshold}: {pattern_count} frequent patterns')


        # transform canonical string format to json
        for threshold in range(maxThreshold, minThreshold-1, -1):
            filtered_pattern_file = os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{threshold}.patterns')
            filtered_json_file = os.path.join(snippetsPath, dataSet, graph_file[:-6] + f'_t{threshold}.json')
            try:
                with open(filtered_pattern_file, 'r') as f_filtered_patterns:
                    with open(filtered_json_file, 'w') as f_filtered_json:
                        json_data = cString2json.parseCStringFileUpToSizePatterns(f_filtered_patterns, patternSize=maxPatternSize)
                        f_filtered_json.write(json_data)
            except EnvironmentError:
                # this might happen if a certain threshold resulted in no frequent patterns and is OK
                pass


# %% Training of classifiers. For later selection of best candidate learners, run xval on train to estimate generalization

def dsf_transform(snippets_file, X):
    with open(snippets_file, 'r') as f_decision_snippets:

        # load decision snippets and create decision snippet features
        frequentpatterns = json.load(f_decision_snippets)
        dsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(
            map(lambda x: x['pattern'], frequentpatterns))
        fts = dsf.fit_transform(X)

        # transform to onehot encodings for subsequent processing by linear methods
        categories = dsf.get_categories()
        fts_onehot_sparse = OneHotEncoder(
            categories=categories).fit_transform(fts)
        fts_onehot = fts_onehot_sparse.toarray()

        return fts_onehot

if run_training:
    results_list = list()

    # save results list
    if not os.path.exists(os.path.join(resultsPath, dataSet)):
        os.makedirs(os.path.join(resultsPath, dataSet))

    def train_model_on_top(model, fts_onehot, Y_train, scoring_function, model_name, descriptor, scaling=False):

        if scaling:
            model = Pipeline([('scaler', StandardScaler()), (model_name, model)])

        fts_onehot_nb_cv_score = cross_val_score(model, fts_onehot, Y_train, cv=5, scoring=scoring_function)
        # dsf_time = time.time() - start_time
        # print(fts_onehot_nb_cv_score)
        dsf_score = fts_onehot_nb_cv_score.mean()
        dsf_std = fts_onehot_nb_cv_score.std()
        print(f'{model_name} {descriptor} {dsf_score} +- {dsf_std}')
        model.fit(fts_onehot, Y_train)
        return dsf_score, model, fts_onehot_nb_cv_score

    # train several models on the various decision snippet features
    # store all xval results on traning data in a list
    for graph_file in filter(lambda x: x.endswith('.json'), os.listdir(os.path.join(snippetsPath, dataSet))):
        
        # get Decision Snippet Features
        fts_onehot = dsf_transform(os.path.join(snippetsPath, dataSet, graph_file), X_train)
       
        # train models
        for model_type, model_class in learners.items():
            xval_score, learner_model, xval_results = train_model_on_top(model_class(**learners_parameters[model_type]), fts_onehot, Y_train, scoring_function, model_type, graph_file)
            results_list.append((xval_score, model_type, graph_file, learner_model, xval_results))
            # cleanup
            xval_score, learner_model, xval_results = None, None, None

        # dump after each decision snippet
        with open(os.path.join(resultsPath, dataSet, "training_xval.pkl"), 'wb') as f_pickle:
            pickle.dump(results_list, f_pickle)

# %% Find, for each learner, the best decision snippet features on training data (the proposed model) and evaluate its performance on test data

print('\n\nHERE ARE THE ACCURACIES ON TEST DATA OF THE BEST DECISION SNIPPET FEATURES\n(for each model and each RF depth)\n')
if run_eval:
    with open(os.path.join(resultsPath, dataSet, "training_xval.pkl"), 'rb') as f_pickle:
        results_list = pickle.load(f_pickle)

        unique_forests = map(lambda x: x[:-5] + '_', filter(lambda x: x.endswith('.json'), os.listdir(os.path.join(forestsPath, dataSet))))

        print(f'best_snippets model_type train_xval_acc test_acc n_features')
        for forest in unique_forests:
            for model_type in learners:
                # print('processing', model_type, forest)
                best_score = 0.
                scores = list()
                labels = list()
                for result in filter(lambda x: x[2].startswith(forest), filter(lambda x: x[1] == model_type, results_list)):
                    # store run with max score
                    if result[0] > best_score:
                        best_result = result
                        best_score = result[0]
                    scores.append(result[0])
                    labels.append(result[2])

                # evaluate on test
                graph_file = best_result[2]
                fts_onehot = dsf_transform(os.path.join(snippetsPath, dataSet, graph_file), X_test)
                pred_test = best_result[3].predict(fts_onehot)
                test_acc = accuracy_score(Y_test, pred_test)

                print(f'{best_result[2]} {model_type} {best_result[0]:.3f} {test_acc:.3f} {fts_onehot.shape[1]}')
