#!/usr/bin/env python

'''
Implementations of the two algorithms from:
An Algorithm for Reducing the Number of Distinct Branching Conditions in a Decision Forest
by Nakamura and Sakurada
'''

#Markus: Alg 1 Min_IntSet
#Manuel: Alg 2 Min_DBN

#Min_IntSet:
#Input: List of 2-Tuples
#Output: 2 Lists: List one contains real numbers, List two conains Lists of indices
import sys
sys.path.append('../arch-forest/code/')
import Forest
import numpy as np
import json

#from code.Forest import *

#from codeTree import *

def Min_IntSet(intervals):
    #preparations: initialize two lists and the number of intervals
    s = []; I = []; n = len(intervals)
    #from now on we follow the pseudocode of Algorithm 1 Min_IntSet (k is not needed)
    i = sorted(range(len(intervals)), key=lambda x: intervals[x][0])    #calculate the indices that would sort the intervals by the values of l_i in ascending order
    t = intervals[i[0]][1]; b = 0
    for j in range(1,n):
        if intervals[i[j]][0] >= t:
            s.append( (intervals[i[j-1]][0] + t) / 2 )
            I.append( [ i[l] for l in range( b, j ) ] )
            t = intervals[i[j]][1]; b = j
        elif( intervals[i[j]][1] < t ):
            t = intervals[i[j]][1]
        #endif
    #endfor
    s.append( (intervals[i[n-1]][0] + t) / 2 ); I.append(  [i[l] for l in range( b, n ) ] )
    return s, I

#operates inplace
def Min_DBN(feature_vectors, decision_forest, sigma):
    d = feature_vectors.shape[1]
    L = []
    for i in range(d):
        L.append([])
    min_l = feature_vectors.min(0)
    max_u = feature_vectors.max(0)
    for j, T in enumerate(decision_forest.trees):
        X_lower, X_higher = precompute_paths(T, feature_vectors)
        for h, N in T.nodes.items():
            i = N.feature
            theta = N.split
            # Filter out leafs
            if (i is None) or ( theta is None ):
                continue
            l, u = get_lu(X_lower[h], X_higher[h], sigma, min_l[i], max_u[i])
            if j ==0 and h==0:
                print(l, u)
            L[i].append((l,u, j, h))

    for i in range(d):
        s, I = Min_IntSet(L[i])
        k = len(I)
        for g in range(k):
            for index in I[g]:
                j, h = L[i][index][2:]
                decision_forest.trees[j].nodes[h].split = s[g]




def precompute_paths(tree, feature_vectors):
    NumNodes = tree.getNumNodes()
    X_lower = []
    X_higher = []
    for i in range(NumNodes):
        X_lower.append([])
        X_higher.append([])
    for x in feature_vectors:
        curNode = tree.head
        while (curNode.prediction is None):
            if x[curNode.feature] <= curNode.split:
                X_lower[curNode.id].append(x[curNode.feature])
                curNode = curNode.leftChild
            else:
                X_higher[curNode.id].append(x[curNode.feature])
                curNode = curNode.rightChild
    return X_lower, X_higher






def get_lu(X_lower, X_higher, sigma, min_l, max_u):
    Threshold = int(np.floor(sigma* (len(X_lower)+len(X_higher))))

    if len(X_lower) < Threshold+1:
        l = -np.inf
        #l = min_l
    else:
        #following could definitely be more efficient by using argpartition
        l = np.sort(X_lower)[-Threshold-1] # max

    if len(X_higher) < Threshold+1:
        u = np.inf
        #u = max_u
    else:
        #following could definitely be more efficient by using argpartition
        u = np.sort(X_higher)[Threshold] #+epsilon #min

    print(X_lower, X_higher, l, u)
    return l, u

    #np.argpartition(vals,)
    #for sigma= Threshold = 0
    #l = max(X_lower) or -inf if X_lower is empty
    #u = min(X_higher)-epsilon, or inf if X_higher is empty

    #for THresh = 1:
    #l = second_max(X_lower) or -inf if |X_lower| < 2
    #u = second_min(X_higher)-epsilon, or inf if X_higher <2
    #...


def post_processing(decision_forest):
    #This function removes all nodes with infinity as split condition and removes all meta information
    #   because it is incorrect as the forest has changed. In addition, the numbering is corrected.
    
    for tree in decision_forest.trees:
        nodeID = 0
        curNode = tree.head
        post_processing_node(nodeID, curNode)
        
def post_processing_node(nodeID, node):
    while(node.split in [-np.inf, np.inf]):     #remove nodes with infinity as branching condition
        if node.split == -np.inf:
            node.fromNode(node.rightChild)
        elif node.split == np.inf:
            node.fromNode(node.leftChild)
    
    node.numSamples = 'null'                    #remove all meta information
    node.id = nodeID                            #renew the numbering
    nodeID += 1
    
    if node.prediction is None:
        node.probLeft = 'null'                  #remove all meta information
        node.probRight = 'null'
        
        nodeID = post_processing_node(nodeID, node.leftChild)
        nodeID = post_processing_node(nodeID, node.rightChild)
        
    return nodeID





def get_feature_vectors():
    #this method returns the feature vectors of the files winequality-red.csv and winequality-white.csv
    #the commands are taken from the trainForest.py file in arch-forest/data/wine-quality
    #red = np.genfromtxt("winequality-red.csv", delimiter=';', skip_header=1)
    #white = np.genfromtxt("winequality-white.csv", delimiter=';', skip_header=1)
    #X = np.vstack((red[:,:-1],white[:,:-1])).astype(dtype=np.float32)
    data = np.genfromtxt("create_test_RF/test_dataset.csv", delimiter=';', skip_header=1)
    X = data[:,:-1]
    return X

if __name__ == '__main__':
    input_file = "create_test_RF/RF_5.json"
    f = Forest.Forest()
    f.fromJSON(input_file)
    print(f.trees[0].getNumNodes())
    t = f.trees[0]
    print(t.nodes[0])
    
    feature_vectors = get_feature_vectors()
    print(feature_vectors[0])
    print(feature_vectors.shape)

    sigma=0.1
    Min_DBN(feature_vectors, f, sigma)
    post_processing(f)
    print(f.pstr())
    
    output_file = input_file.split('.')[0]
    output_file += '_pruned.json'
    
    with open(output_file, 'w') as outfile:
        json.dump(f.pstr(), outfile)


################ Remarks
# Algorthim inplace -> Metadata stimmt nicht mehr
# get_LU can return +- inf
# sort algorithm ineffizient
# supremum epsilon?
# memory efficiency?
# use heaps?
# how to deal with infinity? use max/ min of dimension ? multiply or divide by 2?
# Maybe allow infinity and remove those nodes, where it occures

# varianten:
# 1) +- unendlich setzten -> max double
# 2) knoten mit unendlich rauswerfen

# Einfach ist ok

# gibt's eine klügere variante für thresholds?
# oben schon fehler, dann muss man die die falsch sind unten nicht berücksichtigen..?
# supervised ...?
# 2 Fehler cancceln out -> kein problem
#oben darf man mehr fehlleiten als oben?
# absolutes kriterium? in jedem knoten dürfen nur 100 vectoren umgeleitet werden?

# bias für standard (vorherigen) wert von theta

# Mittelwert der empirischen verteilung nehmen?

# falsche metadaten löschen oder drin lassen.
