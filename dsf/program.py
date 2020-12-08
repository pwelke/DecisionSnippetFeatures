
# %%

from IPython import get_ipython

import csv,operator,sys,os
import numpy as np
import sklearn
import json
import time
from functools import reduce
import subprocess

import DecisionSnippetFeatures

# sys.path.append('./arch-forest/code/')
# import Tree

dataPath = "./arch-forest/data/"
forestsPath = "./tmp/forests/"
resultsPath = "./tmp/snippets/"

dataSet = 'adult'
maxPatternSize = 6
minThreshold = 2
maxThreshold = 25


# %% load data

import ReadData

X_train, Y_train = ReadData.readData(dataSet, 'train', dataPath)
X_train, Y_train = ReadData.readData(dataSet, 'test', dataPath)

print(len(X_train))


# %% create forest data
# sys.path.append('./arch-forest/data/')
from fitModels import fitModels

# fitModels(roundSplit=True, XTrain=X_train, YTrain=Y_train, XTest=None, YTest=None, createTest=False, model_dir=os.path.join(forestsPath, dataSet))


# %% compute decision snippets

import json2graphNoLeafEdgesWithSplitValues

def pattern_frequency_filter(f_patterns, frequency, f_filtered_patterns):
    for line in f_patterns:
        tokens = line.split('\t')
        if int(tokens[0]) >= frequency:
            f_filtered_patterns.write(line)

# translate json to graph files
for json_file in filter(lambda x: x.endswith('.json'), os.listdir(os.path.join(forestsPath, dataSet))):
    
    graph_file = json_file[:-4] + 'graph'
    with open(os.path.join(forestsPath, dataSet, json_file), 'r') as f_in:
        with open(os.path.join(forestsPath, dataSet, graph_file), 'w') as f_out:
            json2graphNoLeafEdgesWithSplitValues.main(f_in, f_out)


# run frequent pattern mining
if not os.path.exists(os.path.join(resultsPath, dataSet)):
    os.makedirs(os.path.join(resultsPath, dataSet))

for graph_file in filter(lambda x: x.endswith('.graph'), os.listdir(os.path.join(forestsPath, dataSet))):
    
    # pattern mining for smallest minThreshold
    print(f"mining {minThreshold}-frequent patterns for {graph_file}")
    pattern_file=os.path.join(resultsPath, dataSet, graph_file[:-6] + f'_t{minThreshold}.patterns')
    feature_file=os.path.join(resultsPath, dataSet, graph_file[:-6] + f'_t{minThreshold}.features')
    log_file=os.path.join(resultsPath, dataSet, graph_file[:-6] + f'_t{minThreshold}.logs')

    args = ['./lwgr', '-erootedTrees', '-mbfs', f'-t{minThreshold}', f'-p{maxPatternSize}', 
            f'-o{pattern_file}', os.path.join(forestsPath, dataSet, graph_file)]
    with open(feature_file, 'w') as f_out:
        with open(log_file, 'w') as f_err:        
            subprocess.run(args, stdout=f_out, stderr=f_err)

    # filtering of patterns for larger thresholds
    print(f"filtering less frequent patterns for {graph_file}")
    for threshold in range(maxThreshold, minThreshold, -1):
        filtered_pattern_file=os.path.join(resultsPath, dataSet, graph_file[:-6] + f'_t{threshold}.patterns')
        with open(pattern_file, 'r') as f_patterns:
            with open(filtered_pattern_file, 'w') as f_filtered_patterns:
                pattern_frequency_filter(f_patterns, threshold, f_filtered_patterns)



# %% xval on decision tree

# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.utils.estimator_checks import check_estimator
# from sklearn.metrics import accuracy_score

# start_time = time.time()
# model = DecisionTreeClassifier(max_depth=10)
# dt_cv_score = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
# dt_time = time.time() - start_time
# dt_score = dt_cv_score.mean()
# print('DT: '+str(dt_score))
# mymodel=Tree.Tree()
# mymodel.fromSKLearn(model)
# print(mymodel.str())



# %%
# load one of the readily available decision trees
f = open(dataPath+dataSet+'/text/DT_5.json')
dt = json.load(f)
f.close()


# %%
# this creates a transformation of the dataset, assigning each example the vertex id of the leaf vertex 
# the example turns out at. So far, so boring
fts = DecisionSnippetFeatures.FrequentSubtreeFeatures(dt).fit_transform(X[:20, :])
print(fts)

# %% [markdown]
# # Something New, Something Interesting
# 
# Let's get funky.
# 
# For this, we need
# 
# - [ ] Frequent Patterns with split values
#     - [x] New Data transformator JSON -> GRAPH
#     - [x] Transform to GRAPH
#     - [x] Mining frequent patterns ('Initial Rooted Frequent Subtree Mining (without embedding computation) -- With Split Values in Labels.ipynb')
#     - [x] New Data transformator CSTRING -> JSON (cString2json.py updated)
#     - [x] Transform to JSON ('Find All Occurrences of All Frequent Patterns of Size up to 6.ipynb')
#     - [ ] fix missing member problem in SubtreeFeatures

# %%
rootedFrequentTrees = "RF_10_t16"
f = open(frequentTreesPath+dataSet+'/WithLeafEdges/leq6/'+rootedFrequentTrees+'.json')
frequentpatterns = json.load(f)
f.close()


# %%
get_ipython().run_line_magic('time', "dsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(map(lambda x: x['pattern'], frequentpatterns[-100:]))")


# %%
get_ipython().run_line_magic('time', 'fts = dsf.fit_transform(X_train)')


# %%
from sklearn.preprocessing import OneHotEncoder
get_ipython().run_line_magic('time', 'fts_onehot = OneHotEncoder(n_values=dsf.get_n_values()).fit_transform(fts)')


# %%
print(X_train.shape)
print(fts_onehot.shape)
print(fts.shape)
print(Y_train.shape)

# %% [markdown]
# # Classification Performance of Decision Tree Snippet Features vs. Normal Features 

# %%
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import accuracy_score

# %% [markdown]
# ### Linear Regression

# %%

model = LinearRegression()

get_ipython().run_line_magic('time', "fts_onehot_cv_score = cross_val_score(model, fts_onehot, Y_train, cv=5, scoring='accuracy')")
#model.fit(fts_onehot,Y_train)
#fts_onehot_cv_score = model.score(X_test,Y_test)
print(fts_onehot_cv_score)
#print(fts_onehot_cv_score.mean())
#print(Y)
#print(model.predict(X))
'''
model = LinearRegression()
model.fit(X_train,Y_train)
normalfeatures_cv_score = model.score(X_test,Y_test)
print(normalfeatures_cv_score)

%time normalfeatures_cv_score = cross_val_score(model, X, Y, cv=5, scoring='r2')
print(normalfeatures_cv_score)
print(normalfeatures_cv_score.mean())
#, scoring='neg_mean_squared_error'

model = DecisionTreeClassifier(max_depth=15)
%time normalfeatures_dt_cv_score = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
print(normalfeatures_dt_cv_score)

model = RandomForestClassifier(max_depth=15, n_estimators=100)
%time normalfeatures_rf_cv_score = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
print(normalfeatures_rf_cv_score)
#f1

with open(resultsPath+dataSet+'/'+rootedFrequentTrees+'.txt', 'w') as f:
        f.write(fts_onehot_cv_score.mean()+'/n'+normalfeatures_cv_score.mean()+'/n'+normalfeatures_dt_cv_score+'/n'+normalfeatures_rf_cv_score)
f.close()'''


# %%
model = LinearRegression()
get_ipython().run_line_magic('time', "normalfeatures_cv_score = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')")
print(normalfeatures_cv_score)

# %% [markdown]
# ### Naive Bayes

# %%
# Classification
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import accuracy_score
    
results_list = []
time_list = []
rf_depth = 5
scoring_function = 'accuracy'
pattern_max_size=6
variant = 'NoLeafEdges'

start_time = time.time()
model = GaussianNB()
normalfeatures_nb_cv_score = cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring_function)
normal_time = time.time() - start_time
normal_score = normalfeatures_nb_cv_score.mean()
print('normal: '+str(normal_score))

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
    start_time = time.time()
    rootedFrequentTrees = "RF_"+str(rf_depth)+"_t"+str(frequency)
    f = open(frequentTreesPath+dataSet+'/'+variant+'/leq'+str(pattern_max_size)+'/'+rootedFrequentTrees+'.json')
    frequentpatterns = json.load(f)
    f.close()

    dsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(map(lambda x: x['pattern'], frequentpatterns[-100:]))
    

    fts = dsf.fit_transform(X_train)

    from sklearn.preprocessing import OneHotEncoder
    fts_onehot = OneHotEncoder(n_values=dsf.get_n_values()).fit_transform(fts)

    



    model = GaussianNB()
    fts_onehot_nb_cv_score = cross_val_score(model, fts_onehot.toarray(), Y_train, cv=5, scoring=scoring_function)
    dsf_time = time.time() - start_time
    print(fts_onehot_nb_cv_score)
    dsf_score = fts_onehot_nb_cv_score.mean()
    print('t'+str(frequency )+'_DSF: '+str(dsf_score))
    
    
    
    result = "t"+str(frequency)+","+str(dsf_score)+","+str(normal_score)+","+str(dt_score)+","+str(rf_score)+",\n"
    results_list.append(result)
    times = "t"+str(frequency)+","+str(dsf_time)+","+str(normal_time)+","+str(dt_time)+","+str(rf_time)+",\n"
    time_list.append(times)

f= open(resultsPath+dataSet+'/Results_'+variant+'/leq'+str(pattern_max_size)+'/'+'RF_'+str(rf_depth)+'_'+scoring_function+'.csv',"a")
for line in results_list:
    f.write(line)
f.close()

f= open(resultsPath+dataSet+'/Results_'+variant+'/leq'+str(pattern_max_size)+'/'+'RF_'+str(rf_depth)+'_'+scoring_function+'_time.csv',"a")
for line in time_list:
    f.write(line)
f.close()


# %%
# Regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import accuracy_score
    
results_list = []
time_list = []
rf_depth = 20
scoring_function_Reg = 'neg_mean_squared_error'
scoring_function_Class = 'accuracy'
pattern_max_size=6

start_time = time.time()
model = LinearRegression()
normalfeatures_nb_cv_score = cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring_function_Reg)
normal_time = time.time() - start_time
normal_score = normalfeatures_nb_cv_score.mean()
print('normal: '+str(normal_score))

start_time = time.time()
model = DecisionTreeClassifier(max_depth=rf_depth)
dt_cv_score = cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring_function_Class)
dt_time = time.time() - start_time
dt_score = dt_cv_score.mean()
print('DT: '+str(dt_score))
    
start_time = time.time()    
model = RandomForestClassifier(max_depth=rf_depth, n_estimators=100)
rf_cv_score = cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring_function_Class)
rf_time = time.time() - start_time
rf_score = rf_cv_score.mean()
print('RF: '+str(rf_score))

for frequency in range(2,26):
    start_time = time.time()
    rootedFrequentTrees = "RF_"+str(rf_depth)+"_t"+str(frequency)
    f = open(frequentTreesPath+dataSet+'/WithLeafEdges/leq'+str(pattern_max_size)+'/'+rootedFrequentTrees+'.json')
    frequentpatterns = json.load(f)
    f.close()

    dsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(map(lambda x: x['pattern'], frequentpatterns[-100:]))
    

    fts = dsf.fit_transform(X_train)

    from sklearn.preprocessing import OneHotEncoder
    fts_onehot = OneHotEncoder(n_values=dsf.get_n_values()).fit_transform(fts)

    



    model = LinearRegression()
    fts_onehot_nb_cv_score = cross_val_score(model, fts_onehot.toarray(), Y_train, cv=5, scoring=scoring_function_Reg)
    dsf_time = time.time() - start_time
    print(fts_onehot_nb_cv_score)
    dsf_score = fts_onehot_nb_cv_score.mean()
    print('t'+str(frequency )+'_DSF: '+str(dsf_score))
    
    
    
    result = "t"+str(frequency)+","+str(dsf_score)+","+str(normal_score)+","+str(dt_score)+","+str(rf_score)+",\n"
    results_list.append(result)
    times = "t"+str(frequency)+","+str(dsf_time)+","+str(normal_time)+","+str(dt_time)+","+str(rf_time)+",\n"
    time_list.append(times)

f= open(resultsPath+dataSet+'/leq'+str(pattern_max_size)+'/'+'RF_'+str(rf_depth)+'_'+scoring_function+'.csv',"a")
for line in results_list:
    f.write(line)
f.close()

f= open(resultsPath+dataSet+'/leq'+str(pattern_max_size)+'/'+'RF_'+str(rf_depth)+'_'+scoring_function+'_time.csv',"a")
for line in time_list:
    f.write(line)
f.close()


# %%
model = GaussianNB()
get_ipython().run_line_magic('time', "normalfeatures_nb_cv_score = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')")
print(normalfeatures_nb_cv_score)
print(normalfeatures_nb_cv_score.mean())


# %%
#with open(resultsPath+dataSet+'/'+rootedFrequentTrees+'.txt', 'w+') as f:
#        f.write('test2')
#f.close()

f= open(resultsPath+dataSet+'/'+rootedFrequentTrees+'.txt',"a")
f.write('test3')
f.close()
#model.fit(fts_onehot.toarray(),Y_train)
#model.predict(X_test[722].reshape(-1, 1))

#print(model.score(X_test,Y_test))

# %% [markdown]
# ### Thresholded Linear Regression

# %%
class LinRegClassifier(LinearRegression):
    def __init__(self, threshold=0.5):
        super(LinRegClassifier, self).__init__()
        self.threshold = threshold
    
    def predict(self, X):
        p = super(LinRegClassifier, self).predict(X)
        return (p > self.threshold)

check_estimator(LinRegClassifier)


# %%
model = LinRegClassifier()
get_ipython().run_line_magic('time', "fts_onehot_nb_cv_score = cross_val_score(model, fts_onehot, Y, cv=5, scoring='f1')")
print(fts_onehot_nb_cv_score)


# %%
model = LinRegClassifier()
get_ipython().run_line_magic('time', "normalfeatures_nb_cv_score = cross_val_score(model, X, Y, cv=5, scoring='f1')")
print(normalfeatures_nb_cv_score)

# %% [markdown]
# ## Comparison to DT and RF on Train

# %%
model = DecisionTreeClassifier(max_depth=15)
get_ipython().run_line_magic('time', "normalfeatures_nb_cv_score = cross_val_score(model, X, Y, cv=5, scoring='f1')")
print(normalfeatures_nb_cv_score)


# %%
model = RandomForestClassifier(max_depth=15, n_estimators=100)
get_ipython().run_line_magic('time', "normalfeatures_nb_cv_score = cross_val_score(model, X, Y, cv=5, scoring='f1')")
print(normalfeatures_nb_cv_score)
#f1

# %% [markdown]
# ## Comparison of LinearRegression on DT features -> Should be more or less identical

# %%
# load one of the readily available decision trees
f = open('/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/arch-forest/data/adult/text/DT_5.json')
dt = json.load(f)
f.close()
dt_fsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(dt)
dt_fts = dt_fsf.fit_transform(X)
dt_fts_onehot = OneHotEncoder(n_values=dt_fsf.get_n_values()).fit_transform(dt_fts)
print(dt_fts_onehot.shape)


# %%
from sklearn.base import BaseEstimator, ClassifierMixin
class UntrainableDTClassifier(BaseEstimator):
    def __init__(self, dt):
        super(UntrainableDTClassifier, self).__init__()
        self.decisionTreeModel = DecisionSnippetFeatures.FrequentSubtreeFeatures(dt)

        tree = self.decisionTreeModel.patterns[0]
        self.linreg_weights = np.zeros(self.decisionTreeModel.get_n_values())
        for i in range(self.decisionTreeModel.get_n_values()):
            try:
                self.linreg_weights[i] = tree.nodes[i].prediction[0]
            except TypeError:
                self.linreg_weights[i] = 0
        
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.dot(X, self.linreg_weights)

check_estimator(UntrainableDTClassifier)
    
    
model = UntrainableDTClassifier(dt)
print(dt_fts_onehot.shape)
get_ipython().run_line_magic('time', '')
dtfeatures_lr_cv_score = cross_val_score(model, dt_fts_onehot, Y, cv=5, scoring='f1')
print(dtfeatures_lr_cv_score)


# %%
a = np.array([1,2,3])
b = np.array([[1,2,3], [1,2,3]])
print(a)
print(b)
print(np.dot(b,a))

# %% [markdown]
# ## Comparison of NaiveBayes on DT features -> Should be more or less identical

# %%
# load one of the readily available decision trees
f = open('/home/falkhoury/Study/Lab/Project/frequentTreesInRandomForests/arch-forest/data/adult/text/DT_5.json')
dt = json.load(f)
f.close()
dt_fsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(dt)
dt_fts = dt_fsf.fit_transform(X)
# dt_fts_onehot = OneHotEncoder(n_values=dt_fsf.get_n_values()).fit_transform(fts)
print(dt_fsf.get_n_values())


# %%
model = GaussianNB()
get_ipython().run_line_magic('time', "dtfeatures_nb_cv_score = cross_val_score(model, dt_fts, Y, cv=5, scoring='neg_mean_squared_error')")
print(dtfeatures_nb_cv_score)


# %%



