import json
from collections import Counter
import subprocess
import tempfile

# import from relative path
from SubtreeSelection.findEmbeddings import annotateEmbeddings
from SubtreeSelection.findNonoverlappingEmbeddings import annotateNonoverlappingEmbeddingsBottomUp, annotateNonoverlappingEmbeddingsTopDown
from SubtreeSelection.cString2json import parseCStringFileUpToSizePatterns




class PatternStatistics():
    """Compute Statistics over all embeddings of a set of rooted subtress in a random forest given a certain weightFunction.

    Subclasses of this class provide constructors to handle different input formats.

    :param weightFunction must either be a string that appears in PatternStatistics.weightFunctions or a function that must accept
    - a pattern of the form (id, tree) where id is an int and tree is a tree in a partial Buschjäger et al. format.
    - a transaction vertex (the root of the embedding being currently weighted)
    as input and
    - output an int.

    :param overlap defines how to select a set of embeddings for each pattern. Are they allowed to overlap?

    """

    transactions = None
    counter = None
    weightFunction = None
    patternDict = None
    overlap = None

    def __init__(self, weightFunction, overlap):
        if isinstance(weightFunction, str):
            self.weightFunction = self.weightFunctions[weightFunction]
        else:
            self.weightFunction = weightFunction

        if isinstance(overlap, str):
            self.overlap = self.overlapDict[overlap]
        else:
            raise ValueError('overlap must be str, but is ' + str(overlap))

    overlapDict = {
        # allow overlapping embeddings (i.e., consider all embeddings)
        'yes' : annotateEmbeddings,

        # bottom up greedy selection of nonoverlapping embeddings; maximizes number of nonoverlapping embeddings
        'bottom_up' : annotateNonoverlappingEmbeddingsBottomUp,

        # top down greedy selection of nonoverlapping embeddins; maximizes number of training examples seen by embeddings
        'top_down' : annotateNonoverlappingEmbeddingsTopDown
    }

    weightFunctions = {
        # weight each embedding with the data support of its root node, if the pattern has more than one vertex
        'data_support' : lambda pattern, transaction: transaction['numSamples'] if len(pattern[1]) > 1 else 0,

        # each embedding of a pattern equally counts for 1
        'frequency' : lambda p,t : 1,

        # this weight function tells us, how much vertices we could save in the RF by contracting each embedding of a pattern into a single vertex:
        'single_node_compression' : lambda p,t : len(p[1]) - 1
    }

    def annotateTransactions(self, patterns):
        '''This function is used by some subclasses to find and store the embeddings of the patterns in the transactions.
        Either all embeddings are found, regardless whether they overlap, or a maximal set of nonoverlapping embeddings
        is found by a bottom up algorithm. '''
        self.overlap(patterns, self.transactions)


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
    """Compute weights of patterns (stored in patternFile) based on their embeddings in a random forest
    (where the embeddings are already stored together with the random forest in transactionFile)
    and output the most important / frequent / whatever patterns prettily.

    See the base class for info on the possible weight functions.
    See  __init__(...) for info on the input file formats.
    """

    def __init__(self, patternFile, embeddingFile, weightFunction, overlap):

        super(PatternStatisticsFromPrecomputed, self).__init__(weightFunction, overlap)

        f = open(patternFile)
        self.patternDict = { pattern['patternid'] : pattern['pattern'] for pattern in json.load(f) }
        f.close()

        f = open(embeddingFile)
        self.transactions = json.load(f)
        f.close()


class PatternStatisticsFromPatternSet(PatternStatistics):
    """Compute weights of patterns (stored in patternFile) in a random forest (stored in transactionFile)
    and output the most important / frequent / whatever patterns prettily.

    See the base class for info on the possible weight functions.
    See  __init__(...) for info on the input file formats.
    """

    def __init__(self, patternFile, transactionFile, weightFunction, overlap):
        """Compute a score for each pattern in pattern file that depends on all embeddings of the pattern in the random
        forest and output the most important / frequent / whatever patterns prettily.
        """

        super(PatternStatisticsFromPatternSet, self).__init__(weightFunction, overlap)

        pf = open(patternFile, 'r')
        patterns = json.load(pf)
        pf.close()

        tf = open(transactionFile, 'r')
        self.transactions = json.load(tf)
        tf.close()

        self.patternDict = { pattern['patternid'] : pattern['pattern'] for pattern in patterns }

        self.annotateTransactions(patterns)


class PatternStatisticsWithMining(PatternStatistics):
    """Mine frequent rooted subtrees in a random forest, compute weights of these patterns in the random forest
    and output the most important / frequent / whatever patterns prettily.

    See the base class for info on the possible weight functions.
    See  __init__(...) for info on the input file formats and parameters.
    """

    def __init__(self, transactionFile, weightFunction, overlap, frequencyThreshold=10, maxPatternSize=10, withLeafVertices=True, withSplitValues=False):
        '''Mine patterns, find all embeddings, and compute pattern weights, given a random forest and some parameters.

        :param transactionFile: a json file in Buschjäger et al.s format containing a random forest
        :param weightFunction: see base class PatternStatistics for available options
        :param frequencyThreshold: an integer giving an absolute frequency threshold for mining. I.e., for a pattern to
          be found frequent it must be subgraph isomorphic to at least frequencyThreshold many decision trees in the
          random forest
        :param maxPatternSize: an integer, specifying the maximum number of vertices of patterns
        :param withLeafVertices: boolean. if False, leaf vertices are not considered in the frequent patterns.
          (although, due to implementation, a single frequent 'leaf' pattern will be found.)
        :param withSplitValues: (boolean) if False, then the vertices of the random forest are labeled only with the
          split feature id. If True, then the vertices of the random forest are labeled 'split_feature_id<value'. This
          usually results in a significantly lower number of frequent patterns.
        '''
        super(PatternStatisticsWithMining, self).__init__(weightFunction, overlap)
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
        self.annotateTransactions(patterns)



if __name__ == '__main__':
    transactionFile = '/home/pascal/Documents/Uni_synced/random_forests/forests/adult/text/RF_10.json'
    test = PatternStatisticsWithMining(transactionFile, weightFunction='data_support', overlap=False, frequencyThreshold=10, maxPatternSize=10)
    print(test.most_common_patterns_string())
