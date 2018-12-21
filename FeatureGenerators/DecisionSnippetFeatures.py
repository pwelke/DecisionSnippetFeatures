import csv,operator,sys,os
import numpy as np
import sklearn
import json
from functools import reduce

sys.path.append('../arch-forest/data/adult/')
sys.path.append('../arch-forest/data/')
sys.path.append('../arch-forest/code/')

import trainForest
import Tree


# subclass Sebastians Tree class to have a function like predict that returns the leaf node id

class FeatureGeneratingTree(Tree.Tree):
    
    def __init__(self, pattern):
        super(FeatureGeneratingTree, self).__init__()
        self.fromJSON(pattern)

    def get_features(self, x):
        curNode = self.head

        # walk through the partial decision tree as long as possible
        while(curNode.prediction is None):
            tmp = curNode
            try:
                if (x[curNode.feature] <= curNode.split): 
                    curNode = curNode.leftChild
                else:
                    curNode = curNode.rightChild
            except:
                return tmp.id
            
        return curNode.id
           
    def get_features_batch(self, X):
        return np.vstack([self.get_features(x) for x in X])
    

class OneHotFeatureGeneratingTree(Tree.Tree):
    
    def __init__(self, pattern):
        super(OneHotFeatureGeneratingTree, self).__init__()
        self.fromJSON(pattern)
        self.n_nodes = len(self.nodes.keys())

    def get_features(self, x):
        curNode = self.head

        # walk through the partial decision tree as long as possible
        while(curNode.prediction is None):
            tmp = curNode
            try:
                if (x[curNode.feature] <= curNode.split): 
                    curNode = curNode.leftChild
                else:
                    curNode = curNode.rightChild
            except:
                return tmp.id
         
        features = np.zeros(self.n_nodes)
        features[curNode.id] = 1
        return features
        
        
    def get_features_batch(self, X):
        return np.vstack([self.get_features(x) for x in X])




# we want to build a feature generator for the input data that is based on frequent subtrees of the random forests 
# trained for the data

class FrequentSubtreeFeatures():
    def __init__(self, patterns=None):
        self.patterns = [FeatureGeneratingTree(pattern) for pattern in patterns]
        self.n_features = len(self.patterns)     
    
    def fit(self, X=None, y=None):
        pass
    
    def transform(self, X):
        return np.hstack([pattern.get_features_batch(X) for pattern in self.patterns])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    
class OneHotFrequentSubtreeFeatures():
    def __init__(self, patterns=None):
        self.patterns = [OneHotFeatureGeneratingTree(pattern) for pattern in patterns]
        self.n_features = sum([p.n_nodes for p in self.patterns])       
    
    def fit(self, X=None, y=None):
        pass
    
    def transform(self, X):
        return np.hstack([pattern.get_features_batch(X) for pattern in self.patterns])     
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
