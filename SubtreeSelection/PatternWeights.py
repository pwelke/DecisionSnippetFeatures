import json
from collections import Counter
import subprocess
import tempfile

# import from relative path
from SubtreeSelection.findEmbeddings import annotateEmbeddings
from SubtreeSelection.cString2json import parseCStringFileUpToSizePatterns




class PatternStatistics():
    """Compute Statistics over all embeddings of a set of rooted subtress in a random forest.


    Subclasses of this class provide constructors to handle different input formats.

    """

    transactions = None
    counter = None
    weightFunction = None
    patternDict = None

    def __init__(self, weightFunction):
        if isinstance(weightFunction, str):
            self.weightFunction = self.weightFunctions[weightFunction]
        else:
            self.weightFunction = weightFunction

    weightFunctions = {
        # weight each embedding with the data support of its root node, if the pattern has more than one vertex
        'data_support' : lambda pattern, transaction: transaction['numSamples'] if len(pattern[1]) > 1 else 0,

        # each embedding of a pattern equally counts for 1
        'frequency' : lambda p,t : 1,

        # this weight function tells us, how much vertices we could save in the RF by contracting each embedding of a pattern into a single vertex:
        'single_node_compression' : lambda p,t : 1 if len(p[1]) - 1 > 1 else 0
    }

    def __embeddingStatsRec(self, currentNode):

        for pattern in currentNode['patterns']:
            self.counter[pattern[0]] += self.weightFunction(pattern, currentNode)

        if 'leftChild' in currentNode.keys():
            self.__embeddingStatsRec(currentNode['leftChild'])
        if 'rightChild' in currentNode.keys():
            self.__embeddingStatsRec(currentNode['rightChild'])

    def embeddingStats(self):
        '''Apply weightfunction to all embeddings of each pattern in each transaction and return, for each pattern,
        the sum of gains.

        Due to implementation, weightfunction must
        - take a pattern and a transaction vertex (the root of the embedding being currently weighted) as input and
        - output an int.
        '''
        if self.counter == None:
            self.counter = Counter()
            for transaction in self.transactions:
                self.__embeddingStatsRec(transaction)
            return self.counter

    def most_common_pattern_ids(self, k=10):
        self.embeddingStats()
        return self.counter.most_common(k)

    def most_common_patterns_string(self, k=10):
        self.embeddingStats()
        return [ '{0}: w={1}, p={2}'.format(x[0], x[1], self.patternDict[x[0]]) for x in self.counter.most_common(k) ]

    def most_common_patterns(self, k=10):
        self.embeddingStats()
        return [ {'id' : x[0], 'weight' : x[1], 'pattern' : self.patternDict[x[0]]} for x in self.counter.most_common(k) ]


class PatternStatisticsFromPrecomputed(PatternStatistics):

    def __init__(self, patternFile, embeddingFile, weightFunction):
        """Create an object that weights the embeddings (stored in transactionFile) of the patterns (stored in patternFile)
        and outputs the most important / frequent / whatever patterns prettily.

        weightFunction must either be a string that appears in PatternStatistics.weightFunctions or a function that must accept
        - a pattern of the form (id, tree) where id is an int and tree is a tree in a partial Buschjäger et al. format.
        - a transaction vertex (the root of the embedding being currently weighted)
        as input and
        - output an int.
        """
        super(PatternStatisticsFromPrecomputed, self).__init__(weightFunction)

        f = open(patternFile)
        self.patternDict = { pattern['patternid'] : pattern['pattern'] for pattern in json.load(f) }
        f.close()

        f = open(embeddingFile)
        self.transactions = json.load(f)
        f.close()


class PatternStatisticsFromSkratch(PatternStatistics):

    def __init__(self, patternFile, transactionFile, weightFunction):
        super(PatternStatisticsFromSkratch, self).__init__(weightFunction)

        pf = open(patternFile, 'r')
        patterns = json.load(pf)
        pf.close()

        tf = open(transactionFile, 'r')
        self.transactions = json.load(tf)
        tf.close()

        self.patternDict = { pattern['patternid'] : pattern['pattern'] for pattern in patterns }

        annotateEmbeddings(patterns, self.transactions)


class PatternStatisticsWithMining(PatternStatistics):

    def __init__(self, transactionFile, weightFunction, frequencyThreshold=10, maxPatternSize=10, withLeafVertices=True, withSplitValues=False):
        super(PatternStatisticsWithMining, self).__init__(weightFunction)
        # store transactions in self.transactions
        tf = open(transactionFile, 'r')
        self.transactions = json.load(tf)
        tf.close()
        # convert transactions for mining
        convertedTransactionFile = self.convertTransactions(transactionFile, withLeafVertices, withSplitValues)
        # mine frequent patterns, annotate all embeddings of the patterns in self.transactions, and store patterns in self.patternDict
        self.mineAndStore(convertedTransactionFile, frequencyThreshold, maxPatternSize)

    def convertTransactions(self, transactionFile, withLeafVertices, withSplitValues):
        # choose an appropriate converter from json to graph
        converter = None
        if withLeafVertices:
            if withSplitValues:
                converter = '../json2graphWithLeafEdgesWithSplitValues.py'
            else:
                converter = '../json2graphWithLeafEdges.py'
        else:
            if withSplitValues:
                converter = '../json2graphNoLeafEdgesWithSplitValues.py'
            else:
                converter = '../json2graphNoLeafEdges.py'

        # convert transaction json file to graph format used by lwgr and store it in a temp file
        tmpfile = tempfile.NamedTemporaryFile()
        finished = subprocess.run(['python3', converter, transactionFile], check=True, stdout=tmpfile)
        return tmpfile

    def mineAndStore(self, transactionFile, frequencyThreshold, maxPatternSize):
        # compute frequent patterns using an external c program and store them in a temp file
        tmpfile = tempfile.NamedTemporaryFile()
        finished = subprocess.run(['../lwgr', '-erootedTrees', '-t' + str(frequencyThreshold), '-p' + str(maxPatternSize), '-o' + tmpfile.name, '-f/dev/null', transactionFile.name], check=True)

        # convert patterns from cstring format to json
        newtmphandle = open(tmpfile.name, 'r')
        patterns = json.loads(parseCStringFileUpToSizePatterns(newtmphandle, maxPatternSize))
        self.patternDict = {pattern['patternid']: pattern['pattern'] for pattern in patterns}
        newtmphandle.close()
        tmpfile.close()

        # find all embeddings of the patterns and store these in self.transactions
        annotateEmbeddings(patterns, self.transactions)



if __name__ == '__main__':
    transactionFile = '/home/pascal/Documents/Uni_synced/random_forests/forests/adult/text/RF_10.json'
    test = PatternStatisticsWithMining(transactionFile, weightFunction='data_support', frequencyThreshold=10, maxPatternSize=10)
    print(test.most_common_patterns_string())
