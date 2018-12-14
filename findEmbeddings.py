#!/usr/bin/env python3

'''
Load a set of patterns and a random forest from two json files and 
find all occurrences of the patterns in the transaction trees. The occurrences that
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

def recCheckEmbedding(pattern, transaction, mapping):
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
                    foundLeft = recCheckEmbedding(pattern['leftChild'], transaction['leftChild'], mapping)    
                else:
                    foundLeft = False
                
            if 'rightChild' in pattern.keys():
                if 'rightChild' in transaction.keys():
                    foundRight = recCheckEmbedding(pattern['rightChild'], transaction['rightChild'], mapping)
                else:
                    foundRight = False
                
            if foundLeft and foundRight:
                mapping[pattern['id']] = transaction['id']
                return True
            else:
                return False
            
    # if we are in the mixed case split vertex vs. leaf vertex then we cannot map the vertices on each other
    return False


def checkForEmbedding(pattern, transaction):
    '''For two given root vertices, check whether pattern is a rooted 
    subtree of transaction such that the roots map to each other 
    and return a mapping id->id if so, o/w None'''
    
    mapping = dict()
    if recCheckEmbedding(pattern, transaction, mapping):
        return mapping
    else:
        return None
    
    
def findAllEmbeddings(pattern, patternid, transaction):
    '''Find all embeddings of pattern into transaction and store them in the transaction
    at the positions where the root vertex of the pattern maps to.
    This method expects to be called after initTransactionTreeForEmbeddingStorage()'''
    if 'feature' in transaction.keys():
        if 'leftChild' in transaction.keys():
            findAllEmbeddings(pattern, patternid, transaction['leftChild'])
        if 'rightChild' in transaction.keys():
            findAllEmbeddings(pattern, patternid, transaction['rightChild'])
    
    mapping = checkForEmbedding(pattern, transaction)
    if mapping != None:
        transaction['patterns'].append((patternid, mapping))
        print(transaction['patterns'])

def initTransactionTreeForEmbeddingStorage(transaction):
    '''We want to be able to store all matching patterns and their embeddings at the
    the vertices, where the root of the pattern maps to. Hence, we init some fields 
    in the transaction decision tree.'''

    if 'feature' in transaction.keys():
        if 'leftChild' in transaction.keys():
            initTransactionTreeForEmbeddingStorage(transaction['leftChild'])
        if 'rightChild' in transaction.keys():
            initTransactionTreeForEmbeddingStorage(transaction['rightChild'])
    
    transaction['patterns'] = list()


def loadAndProcess(patternInput, transactionInput, transactionOutput):
    transactions = json.load(transactionInput)
    patterns = json.load(patternInput)
    
    for transaction in transactions:
        initTransactionTreeForEmbeddingStorage(transaction)
        
    for transaction in transactions:
        for pattern in patterns:
            findAllEmbeddings(pattern['pattern'], pattern['patternid'], transaction)
        
    json.dump(transactions, transactionOutput)
    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('You need exactly two arguments: first a pattern file, second a transaction database, both in a certain json format\n')
        sys.exit(1)
    else:
        patternInFile = open(sys.argv[1], 'r')
        transactionInFile = open(sys.argv[2], 'r')
        loadAndProcess(patternInFile, transactionInFile, sys.stdout)
        patternInFile.close()
        transactionInFile.close()
