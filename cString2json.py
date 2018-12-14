#!/usr/bin/env python3

'''Transform the canonical string format that is given by the lwg and lwgr 
programs to a json format that is compatible to the format of Sebastian.
reads from stdin and prints to stdout

usage: cString2json.py leq|eq patternSize < patternFile > jsonFile

leq results in all patterns up to patternSize vertices being converted,
eq results in all patterns of exactly patternSize vertices being converted.'''


import sys


def cString2json(cString):
    '''Pascals canonical string format and the json format used in Dortmund are 
    basically identical (up to ordering, symbols, and general feeling of course ;) ).
    This is a converter that transforms a single tree from cString format to json format 
    (entirely by string maipulation).'''
    
    intermediate = cString.replace('( leftChild', ',"leftChild":{').replace('( rightChild', ',"rightChild":{').replace(')', '}').replace('leaf', '-1 "prediction":[]')
    tokens = intermediate.split(' ')
    
    json = ''
    i = 0
    for t in tokens:
        try:
            feature = int(t)
            if feature != -1:
                s = '"id":' + str(i) + ',"feature":' + t
            else:
                s = '"id":' + str(i) + ','
            json += s
            i += 1
        except ValueError:
            json += t
            
            
    return ('{' + json.rstrip() + '}')


def parseCStringFileFixedSizePatterns(fIn,  patternSize):
	'''Select the patterns with patternSize vertices from the file f
	with filename. f is assumed to be in the format that lwg or lwgr 
	uses to store the frequent patterns.'''

	# here, we count the number of edges in the pattern
	patternSize = patternSize - 1 

	# gives us the patterns of the selected size
	frequentPatterns = filter(lambda line: line.count('(') == patternSize, fIn)

	# splits the strings into fields
	tokens = map(lambda fp: fp.split('\t'), frequentPatterns)

	# gives us only the canonical strings of the patterns and their id
	pairs = map(lambda t: (t[1], t[2]), tokens)

	# transform to json strings
	jsonCStrings = map(lambda pair: '{"patternid":' + pair[0] + ',"pattern":' + cString2json(pair[1]) + '}', pairs)

	# if your memory explodes, feel free to change this line and the output mode of this function
	jsonBlob = '[' + ',\n'.join(jsonCStrings) + ']'

	return jsonBlob


def parseCStringFileUpToSizePatterns(fIn, patternSize):
	'''Select the patterns up to patternSize vertices from the file f
	with filename. f is assumed to be in the format that lwg or lwgr 
	uses to store the frequent patterns.'''

	# here, we count the number of edges in the pattern
	patternSize = patternSize - 1 

	# gives us the patterns of the selected size
	frequentPatterns = filter(lambda line: line.count('(') <= patternSize, fIn)

	# splits the strings into fields
	tokens = map(lambda fp: fp.split('\t'), frequentPatterns)

	# gives us only the canonical strings of the patterns and their id
	pairs = map(lambda t: (t[1], t[2]), tokens)

	# transform to json strings
	jsonCStrings = map(lambda pair: '{"patternid":' + pair[0] + ',"pattern":' + cString2json(pair[1]) + '}', pairs)

	# if your memory explodes, feel free to change this line and the output mode of this function
	jsonBlob = '[' + ',\n'.join(jsonCStrings) + ']'

	return jsonBlob


if __name__ == '__main__':
	if len(sys.argv) != 3:
		sys.stderr.write('You need exactly two arguments: first leq or eq, second an integer.\n')
		sys.exit(1)
	else:
		try:
			knownFlag = False
			if sys.argv[1] == 'leq': 
				result = parseCStringFileUpToSizePatterns(sys.stdin, int(sys.argv[2]))
				knownFlag = True
			if sys.argv[1] == 'eq': 
				result = parseCStringFileFixedSizePatterns(sys.stdin, int(sys.argv[2]))
				knownFlag = True
			
			if not knownFlag:
				sys.stderr.write('First argument must be either leq or eq.\n')
				sys.exit(1)
			
			sys.stdout.write(result)
			sys.exit(0)

		except ValueError:
			sys.stderr.write('Second argument must be an integer.\n')
			sys.exit(1)