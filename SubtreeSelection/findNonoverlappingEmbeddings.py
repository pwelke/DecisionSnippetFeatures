#!/usr/bin/env python3

'''
Load a set of patterns and a random forest from two json files and 
find a maximal set of occurrences of the patterns in the transaction trees. The occurrences that
are found are stored directly in the vertices of the the transaction trees. I.e.,

- the output written to stdout represents the same random forest as the input random forest.
- each vertex $v$ in the output random forest contains a new member variable "patterns", which is 
  a list of pairs (patternid, mapping), where
  - patternid references a pattern in the input pattern file and 
  - mapping is a map {id11 -> id21, id12 -> id22, ...} indicating that the pattern tree corresponding
    to patternid appears in the random forest such that its root is mapped to the vertex $v$

usage: ./findEmbeddings patternFile randomForestFile > randomForestWithEmbeddingInfoFile

'''



import sys
import json

def recCheckForNonoverlappingEmbedding(pattern, patternid, transaction, mapping):

    # if we try to map to a vertex that is already the root of an embedding found before, we stop
    # this should maximize the overall number of embeddings found (if the process works bottom up over
    # the transaction tree.
    if len(transaction['patterns']) > 0:
        if transaction['patterns'][-1][0] == patternid:
            return False

    # check if we are in a leaf vertex in both pattern and transaction
    if 'prediction' in pattern.keys() and 'prediction' in transaction.keys():
        mapping[pattern['id']] = transaction['id']
        return True
    
    # check if we are in a split vertex in both pattern and transaction
    if 'feature' in pattern.keys() and 'feature' in transaction.keys():

        # check if split features match
        if pattern['feature'] == transaction['feature']:
            
            foundLeft = True
            foundRight = True
            if 'leftChild' in pattern.keys():
                if 'leftChild' in transaction.keys():
                    foundLeft = recCheckForNonoverlappingEmbedding(pattern['leftChild'], patternid, transaction['leftChild'], mapping)
                else:
                    foundLeft = False
                
            if 'rightChild' in pattern.keys():
                if 'rightChild' in transaction.keys():
                    foundRight = recCheckForNonoverlappingEmbedding(pattern['rightChild'], patternid, transaction['rightChild'], mapping)
                else:
                    foundRight = False
                
            if foundLeft and foundRight:
                mapping[pattern['id']] = transaction['id']
                return True
            else:
                return False
            
    # if we are in the mixed case split vertex vs. leaf vertex then we cannot map the vertices on each other
    return False


def checkForNonoverlappingEmbeddingBottomUp(pattern, patternid, transaction):
    '''For two given root vertices, check whether pattern is a rooted
    subtree of transaction such that the roots map to each other
    and return a mapping id->id if so, o/w None
    '''

    mapping = dict()
    if recCheckForNonoverlappingEmbedding(pattern, patternid, transaction, mapping):
        return mapping
    else:
        return None


def recCheckForNonoverlappingEmbedding2(pattern, patternid, transaction, mapping):
    # if we try to map to a vertex that is already part of an embedding found before, we stop.
    # This method, together with recMarkEmbedding, allows us to mark nonoverlapping embeddings top down.
    # This should maximize the overall sum of example data seen by the selected embeddings.
    if transaction['currentEmbedding'] == patternid:
        return False


    # check if we are in a leaf vertex in both pattern and transaction
    if 'prediction' in pattern.keys() and 'prediction' in transaction.keys():
        mapping[pattern['id']] = transaction['id']
        return True

    # check if we are in a split vertex in both pattern and transaction
    if 'feature' in pattern.keys() and 'feature' in transaction.keys():

        # check if split features match
        if pattern['feature'] == transaction['feature']:

            foundLeft = True
            foundRight = True
            if 'leftChild' in pattern.keys():
                if 'leftChild' in transaction.keys():
                    foundLeft = recCheckForNonoverlappingEmbedding2(pattern['leftChild'], patternid,
                                                                   transaction['leftChild'], mapping)
                else:
                    foundLeft = False

            if 'rightChild' in pattern.keys():
                if 'rightChild' in transaction.keys():
                    foundRight = recCheckForNonoverlappingEmbedding2(pattern['rightChild'], patternid,
                                                                    transaction['rightChild'], mapping)
                else:
                    foundRight = False

            if foundLeft and foundRight:
                mapping[pattern['id']] = transaction['id']
                return True
            else:
                return False

    # if we are in the mixed case split vertex vs. leaf vertex then we cannot map the vertices on each other
    return False


def recMarkEmbedding(pattern, patternid, transaction):
    # mark transaction vertex as belonging to the current pattern
    transaction['currentEmbedding'] = patternid

    # check if we are in a leaf vertex in both pattern and transaction
    if 'prediction' in pattern.keys() and 'prediction' in transaction.keys():
        return True

    # check if we are in a split vertex in both pattern and transaction
    if 'feature' in pattern.keys() and 'feature' in transaction.keys():

        # check if split features match
        if pattern['feature'] == transaction['feature']:

            foundLeft = True
            foundRight = True
            if 'leftChild' in pattern.keys():
                if 'leftChild' in transaction.keys():
                    foundLeft = recMarkEmbedding(pattern['leftChild'], patternid, transaction['leftChild'])
                else:
                    foundLeft = False

            if 'rightChild' in pattern.keys():
                if 'rightChild' in transaction.keys():
                    foundRight = recMarkEmbedding(pattern['rightChild'], patternid, transaction['rightChild'])
                else:
                    foundRight = False

            if foundLeft and foundRight:
                return True
            else:
                return False

    # if we are in the mixed case split vertex vs. leaf vertex then we cannot map the vertices on each other
    return False


def checkForNonoverlappingEmbeddingTopDown(pattern, patternid, transaction):
    '''For two given root vertices, check whether pattern is a rooted
    subtree of transaction such that the roots map to each other
    and return a mapping id->id if so, o/w None
    '''

    mapping = dict()
    if recCheckForNonoverlappingEmbedding2(pattern, patternid, transaction, mapping):
        if not recMarkEmbedding(pattern, patternid, transaction):
            raise Exception('marking did not work for pattern id ' + str(patternid) + ' and mapping ' + str(mapping))
        return mapping
    else:
        return None


def findNonoverlappingEmbeddingsBottomUp(pattern, patternid, transaction):
    '''Find all embeddings of pattern into transaction and store them in the transaction
    at the positions where the root vertex of the pattern maps to.
    This method expects to be called after initTransactionTreeForEmbeddingStorage()

    It does a bottom up greedy selection of nonoverlapping embeddings. This is guaranteed
    to select a cardinality maximum set of nonoverlapping embeddings in transaction. See
    Masuyama, S.: On the tree packing problem. Discrete Applied Mathematics, 1992, 35, 163-166

    However, this is not guaranteed to maximize arbitrary weight functions over the embeddings.
    In certain cases, a top down evaluation might be more useful (cf. documentation there).

    There must be a smarter way than the following (which is what Masuyama proposes).
    However, this is rather easy to implement:
    We iterate (recursively) over the transaction (decision) tree 
    vertices $v$ and check whether there is a rooted subgraph 
    isomorphism from pattern to the transaction mapping the root of 
    pattern to $v$.
    This is decided by (again) recursion over pattern and transaction 
    simultaneously, as long as it fits.

    We'll see whether this is fast enough for our case. Its something 
    along $O(n * p)$ where $n$ and $p$ are the numbers of vertices of 
    transactions and patterns, respectively.'''

    if 'leftChild' in transaction.keys():
        findNonoverlappingEmbeddingsBottomUp(pattern, patternid, transaction['leftChild'])
    if 'rightChild' in transaction.keys():
        findNonoverlappingEmbeddingsBottomUp(pattern, patternid, transaction['rightChild'])

    # it is important to do bottom up processing of the transaction!
    mapping = checkForNonoverlappingEmbeddingBottomUp(pattern, patternid, transaction)

    if mapping != None:
        transaction['patterns'].append((patternid, mapping))

def findNonoverlappingEmbeddingsTopDown(pattern, patternid, transaction):
    '''Find all embeddings of pattern into transaction and store them in the transaction
    at the positions where the root vertex of the pattern maps to.
    This method expects to be called after initTransactionTreeForEmbeddingStorage()

    It does a top down greedy selection of nonoverlapping embeddings. This is guaranteed
    to select a maximum weight set of nonoverlapping embeddings in transaction if
    - the weights of the embeddings can be expressed by a weight function on the root vertices of the
      embeddings in transaction and
    - for all vertices v in transaction the sum of weights of the children of v is smaller or equal to the weight of v.
    (For example, if the weight function is the number of training examples seen by a split/leaf vertex of a decision tree)

    However, this is not guaranteed to maximize arbitrary weight functions over the embeddings.
    In certain cases, a bottom up evaluation might be more useful (cf. documentation there).

    There must be a smarter way than the following (which is what Masuyama proposes).
    However, this is rather easy to implement:
    We iterate (recursively) over the transaction (decision) tree
    vertices $v$ and check whether there is a rooted subgraph
    isomorphism from pattern to the transaction mapping the root of
    pattern to $v$.
    This is decided by (again) recursion over pattern and transaction
    simultaneously, as long as it fits.

    We'll see whether this is fast enough for our case. Its something
    along $O(n * p)$ where $n$ and $p$ are the numbers of vertices of
    transactions and patterns, respectively.'''

    # for some cases, top down evaluation might be useful.
    mapping = checkForNonoverlappingEmbeddingTopDown(pattern, patternid, transaction)
    if mapping != None:
        transaction['patterns'].append((patternid, mapping))

    if 'leftChild' in transaction.keys():
        findNonoverlappingEmbeddingsTopDown(pattern, patternid, transaction['leftChild'])
    if 'rightChild' in transaction.keys():
        findNonoverlappingEmbeddingsTopDown(pattern, patternid, transaction['rightChild'])


def initTransactionTreeForEmbeddingStorage(transaction):
    '''We want to be able to store all matching patterns and their embeddings at the
    the vertices, where the root of the pattern maps to. Hence, we init some fields 
    in the transaction decision tree.'''

    if 'leftChild' in transaction.keys():
        initTransactionTreeForEmbeddingStorage(transaction['leftChild'])
    if 'rightChild' in transaction.keys():
        initTransactionTreeForEmbeddingStorage(transaction['rightChild'])
    
    transaction['patterns'] = list()
    transaction['currentEmbedding'] = None


def annotateNonoverlappingEmbeddingsBottomUp(patterns, transactions):
    
    for transaction in transactions:
        initTransactionTreeForEmbeddingStorage(transaction)
        
    for transaction in transactions:
        for pattern in patterns:
            findNonoverlappingEmbeddingsBottomUp(pattern['pattern'], pattern['patternid'], transaction)


def annotateNonoverlappingEmbeddingsTopDown(patterns, transactions):
    for transaction in transactions:
        initTransactionTreeForEmbeddingStorage(transaction)

    for transaction in transactions:
        for pattern in patterns:
            findNonoverlappingEmbeddingsTopDown(pattern['pattern'], pattern['patternid'], transaction)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('You need exactly two arguments: first a pattern file, second a transaction database, both in a certain json format\n')
        sys.exit(1)

    transactionInFile = open(sys.argv[2], 'r')
    transactions = json.load(transactionInFile)
    transactionInFile.close()

    patternInFile = open(sys.argv[1], 'r')
    patterns = json.load(patternInFile)
    patternInFile.close()

    annotateNonoverlappingEmbeddingsBottomUp(patterns, transactions)

    json.dump(transactions, sys.stdout)

