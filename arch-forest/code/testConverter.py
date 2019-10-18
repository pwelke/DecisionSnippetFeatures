#!/usr/bin/env python3

import csv,operator,sys
import numpy as np
import os.path
import pickle
import sklearn
import json
import gc
import objgraph

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
import timeit
from sklearn.externals import joblib

np.set_printoptions(threshold=np.inf)

import sys
sys.setrecursionlimit(20000)
#sys.path.append('../code/')

import Forest
from ForestConverter import *
from NativeTreeConverter import *
from IfTreeConverter import *
from MixConverter import *

# A template to test the generated code
testCodeTemplate = """#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <cassert>
#include <tuple>
#include <chrono>

{headers}

void readCSV({feature_t} * XTest, unsigned int * YTest) {
	std::string line;
	std::ifstream file("{test_file}");
	unsigned int xCnt = 0;
	unsigned int yCnt = 0;
	unsigned int lineCnt = 0;

	if (file.is_open()) {
		while ( std::getline(file,line)) {
			if ( line.size() > 0) {
				std::stringstream ss(line);
				std::string entry;
				unsigned int first = true;

				while( std::getline(ss, entry,',') ) {
					if (entry.size() > 0) {
						if (first) {
							YTest[yCnt++] = (unsigned int) atoi(entry.c_str());
							first = false;
						} else {
							//XTest[xCnt++] = ({feature_t}) atoi(entry.c_str());
							XTest[xCnt++] = ({feature_t}) atof(entry.c_str());
						}
					}
				}
				lineCnt++;
				if( lineCnt > {N} ) {
					break;
				}
			}
		}
		file.close();
	}

}

int main(int argc, char const *argv[]) {

	//std :: cout << "=== NEW PERFORMANCE TEST ===" << std :: endl;
	//std :: cout << "Testing dimension:\t" << {DIM} << std :: endl;
	//std :: cout << "Feature type:\t "<< "{feature_t}" << std :: endl;
	//std :: cout << "Testing instances:\t" << {N} << std :: endl << std :: endl;
	//std :: cout << "Loading testing data..." << std :: endl;


	{allocMemory}
	readCSV(XTest,YTest);

	{measurmentCode}
	{freeMemory}

	return 1;
}
"""

measurmentCodeTemplate = """
	/* Burn-in phase to minimize cache-effect and check if data-set is okay */
	for (unsigned int i = 0; i < 2; ++i) {
		unsigned int acc = 0;
		for (unsigned int j = 0; j < {N}; ++j) {
			unsigned int pred = {namespace}_predict(&XTest[{DIM}*j]);
			acc += (pred == YTest[j]);
		}

		// SKLearn uses a weighted majority vote, whereas we use a "normal" majority vote
		// Therefore, we may not match the accuracy of SKlearn perfectly!
		if (acc != {target_acc}) {
			std :: cout << "Target accuracy was not met!" << std :: endl;
			std :: cout << "\t target: {target_acc}" << std :: endl;
			std :: cout << "\t current:" << acc << std :: endl;
			//return 1;
		}
	}

	std::vector<float> runtimes;
	std::vector<unsigned int> accuracies;
	unsigned int pred;
	for (unsigned int i = 0; i < {num_repetitions}; ++i) {
		unsigned int acc = 0;
		auto start = std::chrono::high_resolution_clock::now();
		for (unsigned int j = 0; j < {N}; ++j) {
			pred = {namespace}_predict(&XTest[{DIM}*j]);
			acc += (pred == YTest[j]);
			//acc += pred;
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		runtimes.push_back((float) (duration.count() / {N}.0f));
	}

	// Something close to welfords algorithm to estimate variance and mean on the fly
	float avg = 0.0f;
	float var = 0.0f;
	unsigned int cnt = 0;
	for (auto d : runtimes) {
		cnt++;
		float delta = d - avg;
		avg = avg + delta / cnt;
		float delta2 = d - avg;
		var = var + delta*delta2;
	}

	//std :: cout << "Runtime per element (ms): " << avg << " ( " << var / (cnt - 1) << " )" <<std :: endl;
	std :: cout << avg << "," << var / (cnt - 1) << std :: endl;
"""

def writeFiles(basepath, basename, header, cpp):
	if header is not None:
		with open(basepath + basename + ".h",'w') as code_file:
			code_file.write(header)

	if cpp is not None:
		with open(basepath + basename + ".cpp",'w') as code_file:
			code_file.write(cpp)

def writeTestFiles(outPath, namespace, header, dim, N, featureType, testFile, targetAcc, reps):
	allocMemory = "{feature_t} * XTest = new {feature_t}[{DIM}*{N}];\n \tunsigned int * YTest = new unsigned int[{N}];"
	freeMemory = "delete[] XTest;\n \tdelete[] YTest;"

	measurmentCode = measurmentCodeTemplate.replace("{namespace}", namespace).replace("{target_acc}", str(targetAcc)).replace("{num_repetitions}", str(reps))

	testCode = testCodeTemplate.replace("{headers}", "#include \"" + header + "\"") \
							   .replace("{allocMemory}", allocMemory) \
							   .replace("{freeMemory}", freeMemory) \
							   .replace("{measurmentCode}",measurmentCode) \
							   .replace("{feature_t}", str(featureType)) \
							   .replace("{N}", str(N)) \
							   .replace("{DIM}", str(dim)) \
							   .replace("{test_file}", testFile)

	with open(outPath + namespace + ".cpp",'w') as code_file:
		code_file.write(testCode)

def generateClassifier(outPath, targetAcc, DIM, N,converter, namespace, featureType, forest, testFile, reps):
	#print("GETTING THE CODE")
	headerCode, cppCode = converter.getCode(forest)
	cppCode = "#include \"" + namespace + ".h\"\n" + cppCode
	writeFiles(outPath, namespace, headerCode, cppCode)
	writeTestFiles(outPath+"test", namespace, namespace + ".h", DIM, N, featureType, testFile, targetAcc, reps)

def getFeatureType(X):
	containsFloat = False
	for x in X:
		for xi in x:
			if isinstance(xi, np.float32):
				containsFloat = True
				break

	if containsFloat:
		dataType = "float"
	else:
		lower = np.min(X)
		upper = np.max(X)
		bitUsed = 0
		if lower > 0:
			prefix = "unsigned"
			maxVal = upper
		else:
			prefix = ""
			bitUser = 1
			maxVal = max(-lower, upper)

		bit = int(np.log2(maxVal) + 1 if maxVal != 0 else 1)

		if bit <= (8-bitUsed):
			dataType = prefix + " char"
		elif bit <= (16-bitUsed):
			dataType = prefix + " short"
		else:
			dataType = prefix + " int"

	return dataType

def main(argv):
	X = None
	Y = None

	forestPath = "RF_15.json"
	for i in range(10):
		loadedForest = Forest.Forest()
		loadedForest.fromJSON(forestPath)

		if X is None:
			data = np.loadtxt("RF_15.csv", delimiter = ",")

			X = data[:,1:].astype(dtype=np.float32)
			Y = data[:,0]

		clf = joblib.load("RF_15.pkl")
		print("\tComputing target accuracy")
		YPredicted_ = loadedForest.predict_batch(X)
		YPredictedSK = clf.predict(X)
		targetAcc = sum(YPredicted_ == Y)
		#print("\tAccuracy MY:%s" % accuracy_score(Y, YPredicted_))
		print("\tWeighted Majority Vote: %s" % sum(YPredictedSK == Y))
		print("\tStandard Majority Vote: %s" % sum(YPredicted_ == Y))

		featureType = getFeatureType(X)
		dim = len(X[0])
		converter = ForestConverter(OptimizedIFTreeConverter(dim, "OptimizedNodeIfTree", featureType, "intel", "node", 25))
		generateClassifier("./", targetAcc, dim, len(X), converter, "OptimizedNodeIfTree", featureType, loadedForest, "RF_15.csv", 2)

		# loadedForest.cleanup()
		loadedForest = None
		converter = None

		#gc.set_debug(gc.DEBUG_LEAK)
		gc.collect()
		#print(sys.getrefcount(loadedForest))
		#print(sys.getrefcount(converter))

		objgraph.show_most_common_types(limit=20)

if __name__ == "__main__":
   main(sys.argv[1:])
