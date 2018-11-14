## This script creates a graph database from the decision trees in Sebastians json Format as follows:
# - the root vertex of the tree has index 1 (counting starts with 1)
# - each vertex is labeled by its split feature or by 'leaf'
# - each edge is labeled either 'leftChild' or 'rightChild'
# - there are no edges containing 'leaf' vertices
#
# It follows that the connected components resulting from a single decision tree are several isolated vertices labeled 'leaf' 
# and a tree containing all the split vertices.


import json
import sys

def recParse(vertex, vertexLabels, edges):
    if 'feature' in vertex.keys():
        vertexLabels[vertex['id'] + 1] = vertex['feature']
        if 'leftChild' in vertex.keys():
            if 'feature' in vertex['leftChild'].keys():
                edges.append((str(vertex['id'] + 1), str(vertex['leftChild']['id'] + 1), 'leftChild'))
            recParse(vertex['leftChild'], vertexLabels, edges)
        if 'rightChild' in vertex.keys():
            if 'feature' in vertex['rightChild'].keys():
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

