
# coding: utf-8

# # Ideas
# 
# - random forests: are there frequent subtrees in the random forests generated from some data set
# - labels of vertices in the forests can be of two variants:
#     - only the feature on which the split happens
#     - feature < threshold
# - should we keep the leaf vertices (the decisions) or should we skip them?
# - which subtrees are interesting?
#     - each vertex in the tree defines a subtree of all vertices below it
#     - rooted subtrees in the classical sense
#     - unrooted subtrees
#     
# 

## This script creates a graph database from the decision trees in Sebastians json Format as follows:
# - the root vertex of the tree has index 1 (counting starts with 1)
# - each vertex is labeled by its split feature or by 'leaf'
# - each edge is labeled either 'leftChild' or 'rightChild'
# - for each edge in a decision tree there is an edge in the graph database
#
# It follows that the resulting trees in the database correspond 1-1 to the decision trees in the json.
# However, we fiddle with the labels of the vertices and forget, e.g., the threshold on which to split.


import json
import sys

def recParse(vertex, vertexLabels, edges):
    if 'feature' in vertex.keys():
        vertexLabels[vertex['id'] + 1] = vertex['feature']
        if 'leftChild' in vertex.keys():
            edges.append((str(vertex['id'] + 1), str(vertex['leftChild']['id'] + 1), 'leftChild'))
            recParse(vertex['leftChild'], vertexLabels, edges)
        if 'rightChild' in vertex.keys():
            edges.append((str(vertex['id'] + 1), str(vertex['rightChild']['id'] + 1), 'rightChild'))
            recParse(vertex['rightChild'], vertexLabels, edges)
    else:
        vertexLabels[vertex['id'] + 1] = 'leaf'

    return (vertexLabels, edges)

def parseTree(tree):
    vertexLabels = dict()
    edges = list()
    return recParse(tree, vertexLabels, edges)

def transform2GraphDB(vertexLabels, edges, graphCounter, out):
    print(' '.join(['#', str(graphCounter), '0', str(len(vertexLabels)), str(len(edges))]))
    for v in sorted(vertexLabels.keys()):
        out.write(str(vertexLabels[v]) + ' ')
    out.write('\n')
        
    for e in edges:
        out.write(' '.join(e) + ' ')
    out.write('\n')
        
def main(file, out):
    f = open(file)
    j = json.load(f)
    f.close()

    graphCounter = 0
    for tree in j:
        vertexLabels, edges = parseTree(tree)
        transform2GraphDB(vertexLabels, edges, graphCounter, out)
        graphCounter += 1
    print('$')

if __name__ == '__main__':
    main(sys.argv[1], sys.stdout)

