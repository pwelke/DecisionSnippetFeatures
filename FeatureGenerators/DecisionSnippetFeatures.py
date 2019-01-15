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

# Utility functions to ensure compatibility between my frequent trees and Sebastians decision trees 

def _getMaxVertexId(vertex):
    if 'leftChild' in vertex.keys():
        leftMax = _getMaxVertexId(vertex['leftChild'])
    else:
        leftMax = 0
    
    if 'rightChild' in vertex.keys():
        rightMax = _getMaxVertexId(vertex['rightChild'])
    else:
        rightMax = 0

    return max(leftMax, rightMax, vertex['id'])


def _fillMembers(vertex, maxId):
    # ensure that all required members are there and that the tree is (unbalanced) binary
    if 'numSamples' not in vertex.keys():
        vertex['numSamples'] = 0

    # TODO this is just a temporary fix to get it to run. 
    # should be thought through more rigidly...
    if 'feature' in vertex.keys():
        if 'probLeft' not in vertex.keys():
            vertex['probLeft'] = 0
        if 'probRight' not in vertex.keys():
            vertex['probRight'] = 0
        if 'isCategorical' not in vertex.keys():
            vertex['isCategorical'] = False
        if 'feature' not in vertex.keys():
            vertex['feature'] = 0
        if 'split' not in vertex.keys():
            vertex['split'] = 0

        # ensure that split nodes have two children
        if 'leftChild' in vertex.keys():
            maxId = _fillMembers(vertex['leftChild'], maxId)
        else:
            maxId += 1
            vertex['leftChild'] = {'id':maxId, 'numSamples':0, 'prediction':list()}    
        if 'rightChild' in vertex.keys():
            maxId = _fillMembers(vertex['rightChild'], maxId)
        else:
            maxId += 1
            vertex['rightChild'] = {'id':maxId, 'numSamples':0, 'prediction':list()}

    return maxId

def makeProperBinaryDT(vertex):
    maxUsedVertexId = _getMaxVertexId(vertex)
    _fillMembers(vertex, maxUsedVertexId)
    return vertex



# subclass Sebastians Tree class to have a function like predict that returns the leaf node id on which the data maps
# (instead of the prediction given by that node)

class FeatureGeneratingTree(Tree.Tree):
    
    def __init__(self, pattern):
        super(FeatureGeneratingTree, self).__init__()
        self.fromJSON(makeProperBinaryDT(pattern))
        self.n_nodes = len(self.nodes.keys())


    def get_features(self, x):
        curNode = self.head

        # walk through the partial decision tree as long as possible
        while(curNode.prediction == None):            
            if (x[curNode.feature] <= curNode.split): 
                curNode = curNode.leftChild
            else:
                curNode = curNode.rightChild
         
        return curNode.id
           
    def get_features_batch(self, X):
        return np.array([self.get_features(x) for x in X])


# we want to build a feature generator for the input data that is based on frequent subtrees of the random forests 
# trained for the data

class FrequentSubtreeFeatures():
    def __init__(self, patterns=None):
        self.patterns = [FeatureGeneratingTree(pattern) for pattern in patterns]
        self.n_features = len(self.patterns)

    def get_n_values(self):
        ''' To allow OneHotEncoding with a fixed number of features that does not depend on the data, 
        but only on the FeatureGeneratingTrees present in the model, use the following code:

        dsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(map(lambda x: x['pattern'], frequentpatterns[-100:])) 
        fts = dsf.fit_transform(X)
        fts_onehot = OneHotEncoder(n_values=dsf.get_n_values()).fit_transform(fts)
        '''
        return [pattern.n_nodes for pattern in self.patterns]
    
    def fit(self, X=None, y=None):
        pass
    
    def transform(self, X):
        return np.stack([pattern.get_features_batch(X) for pattern in self.patterns]).T

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

