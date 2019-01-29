import json

# import from relative path
from .findEmbeddings import annotateEmbeddings
from collections import Counter


class PatternStatistics():

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
