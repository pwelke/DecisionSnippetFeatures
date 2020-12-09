import csv,operator,sys,os
import numpy as np
import sklearn
import json
from functools import reduce

sys.path.append('../arch-forest/data/adult/')
sys.path.append('../arch-forest/data/')
sys.path.append('../arch-forest/code/')

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

    # TODO this is just a temporary fix to get it to run. should be thought through more thoroughly...
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


class FeatureGeneratingTree(Tree.Tree):
    """A subclass of Sebastian Buschjäger et al.'s Tree class.
    It extends it to have a function like predict that returns the leaf node id on which the data maps
    instead of the prediction given by that node."""
    
    def __init__(self, pattern):
        super(FeatureGeneratingTree, self).__init__()
        self.fromJSON(makeProperBinaryDT(pattern))
        self.n_nodes = len(self.nodes)

    
    def get_features(self, x, output=0):
        ''' to get nodeId set output as 0, to get count of comparisons set output to 1'''
        curNode = self.head
        counter = 0 

        # walk through the (partial) decision tree as long as possible
        while(curNode.prediction == None):
            counter +=1
            if (x[curNode.feature] <= curNode.split): 
                curNode = curNode.leftChild
            else:
                curNode = curNode.rightChild
        if (output == 0):           
            return curNode.id
        else:
            return counter
        #return counter
        #self.x_counter +=1
           
    def get_features_batch(self, X, output=0):
        return np.array([self.get_features(x, output) for x in X])


class FrequentSubtreeFeatures():
    """A feature extraction algorithm that transforms you data point(s) x into a categorical feature space F
    corresponding to a random forest R. Each feature f in F corresponds to a decision tree T in the random forest R
    and each value of f corresponds to a leaf of T. That is, f(x) is the id of the leaf of T in which x would end up in.

    Most likely, you want to transform the features created here to a one-hot encoding. See the documentation of
    get_n_values() for some hints.
    """

    def __init__(self, patterns=None):
        """Init a new Feature Extractor Object corresponding to a random forest given as a list of decision trees as
        parsed json objects in the format used by Sebastian Buschjäger et al.
        ( available via git: git clone git@bitbucket.org:sbuschjaeger/arch-forest.git )

        Mainly used to create Feature Extractors that correspond to sets of frequent rooted subtrees in random forests.
        """
        self.patterns = [FeatureGeneratingTree(pattern) for pattern in patterns]
        self.n_features = len(self.patterns)

    def get_n_values(self):
        """ Return the size of the feature set if the model is based on a single tree
        or a list of sizes of the individual feature sets of all trees in the model if there are more than one tree.

        This method is compatible with sklearn.preprocessing.OneHotEncoder in the following way:

        To allow OneHotEncoding with a fixed number of features that does not depend on the data,
        but only on the FeatureGeneratingTrees present in the model, use the following code:

        dsf = DecisionSnippetFeatures.FrequentSubtreeFeatures(map(lambda x: x['pattern'], frequentpatterns[-100:]))
        fts = dsf.fit_transform(X)
        fts_onehot = OneHotEncoder(n_values=dsf.get_n_values()).fit_transform(fts)
        """
        size_list = [pattern.n_nodes for pattern in self.patterns]
        if len(size_list) == 1:
            return size_list[0]
        else:
            return size_list
    
    def fit(self, X=None, y=None):
        """Nothing to be done. The fitting already happenened during the creation of the random forest/decision tree/
        frequent rooted subtree models."""
        pass
    
    def transform(self, X, output=0):
        """Compute the ids of the leafs of the decision trees that the data points end up in. (default)
        Or compute the number of comparisons made during leaf id inference"""
        return np.stack([pattern.get_features_batch(X, output) for pattern in self.patterns]).T

    def fit_transform(self, X, output=0, y=None):
        """Equivalent to transform(X).
        Compute the ids of the leafs of the decision trees that the data points end up in. (default)
        Or compute the number of comparisons made during leaf id inference"""
        return self.transform(X, output)

