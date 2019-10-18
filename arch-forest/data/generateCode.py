#!/usr/bin/env python3

import csv,operator,sys
import numpy as np
import os.path
import pickle
import sklearn
import json
import gc
#import objgraph

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
sys.path.append('../code/')

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
	std::vector<unsigned int> accuracies;
	/* Burn-in phase to minimize cache-effect and check if data-set is okay */
	for (unsigned int i = 0; i < 2; ++i) {
		unsigned int acc = 0;
		for (unsigned int j = 0; j < {N}; ++j) {
			unsigned int pred = {namespace}_predict(&XTest[{DIM}*j]);
			acc += (pred == YTest[j]);
		}

		// SKLearn uses a weighted majority vote, whereas we use a "normal" majority vote
		// Therefore, we may not match the accuracy of SKlearn perfectly!
		//if (acc != {target_acc}) {
		//	std :: cout << "Target accuracy was not met!" << std :: endl;
		//	std :: cout << "\t target: {target_acc}" << std :: endl;
		//	std :: cout << "\t current:" << acc << std :: endl;
			//return 1;
		//}
		accuracies.push_back(acc);
	}

	std::vector<float> runtimes;
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
		std::chrono::nanoseconds duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

		runtimes.push_back((float) (duration.count() / {N}.0f));
		accuracies.push_back(acc);
	}

	// Something close to welfords algorithm to estimate variance and mean on the fly
	float avg = 0.0f;
	float var = 0.0f;
	float max, min;
	unsigned int cnt = 0;
	for (auto d : runtimes) {
		if (cnt == 0) {
			max = d;
			min = d;
		} else {
			if (max < d) {
				max = d;
			}
			if (min > d) {
				min = d;
			}
		}

		cnt++;
		float delta = d - avg;
		avg = avg + delta / cnt;
		float delta2 = d - avg;
		var = var + delta*delta2;
	}

	//std :: cout << "Runtime per element (ms): " << avg << " ( " << var / (cnt - 1) << " )" <<std :: endl;
	std :: cout << avg << "," << var / (cnt - 1) << "," << min << "," << max << std :: endl;
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

def debug_gc():
	gc.collect()
	# objects = gc.get_objects()
	# objects_id = {}
	# for o in objects:
	# 	objects_id[id(o)] = True

	# verbose = 2

	# for o in gc.get_objects():
	# 	print(o)

def main(argv):
	if len(argv)<1:
		print("Please give a sub-folder / dataset to be used")
		return
	else:
		basepath = argv[0].strip("/")

	if len(argv) < 2:
		print("Please give a target architecture (arm or intel or ppc)")
		return
	else:
		target = argv[1]

		if (target != "intel" and target != "arm" and target != "ppc"):
			print("Did not recognize architecture, ", target)
			print("Please use arm or intel or ppc")
			return

	#if len(argv) < 3:
	if target == "intel":
		setSizes = [25]
		# budgetSizes = [128*1000, 384*1000]
		budgetSizes = [128*1000]
		#setSize = 10 # 5,10,25,50
		#budgetSize = 32*1000 # 16*1000, 32*1000, 64*1000
	else:
		setSizes = [8]
		# setSizes = [8,32]
		budgetSizes = [32*1000]
		# budgetSizes = [32*1000, 64*1000]
			#setSize = 8 # 5,8,20,40
			#budgetSize = 32*1000 # 16*1000, 32*1000, 64*1000
	# else:
	# 	setSize = int(argv[2])
	reps = 50 # 20

	# if len(argv) < 4:
	# 	reps = 20
	# else:
	# 	reps = argv[2]

	if not os.path.exists(basepath + "/cpp"):
		os.makedirs(basepath + "/cpp")

	if not os.path.exists(basepath + "/cpp/" + target):
		os.makedirs(basepath + "/cpp/" + target)

	X = None
	Y = None

	for f in sorted(os.listdir(basepath + "/text/")):
		if f.endswith(".json"):
			name = f.replace(".json","")
			cppPath = basepath + "/cpp/" + target + "/" + name
			print("Generating", cppPath)

			if not os.path.exists(cppPath):
				os.makedirs(cppPath)

			forestPath = basepath + "/text/" + f

			print("\tLoading forest")

			loadedForest = Forest.Forest()
			loadedForest.fromJSON(forestPath)


			if X is None:
				print("\tReading CSV file to compute test accuracy")
				# file = open(basepath + "/test.csv")
				# for row in file:
				# 	entries = row.replace("\n","").split(",")
				# 	Y.append(float(entries[0]))
				# 	x = []
				# 	for e in entries[1:]:
				# 		x.append(float(e))
				# 	X.append(x)

				# X = np.array(X)
				# Y = np.array(Y)

				data = np.loadtxt(basepath + "/test.csv", delimiter = ",")

				X = data[:,1:]
				Y = data[:,0]

				if target == "arm" or target == "ppc":
					numTest = min(len(X),10000)
				else:
					numTest = len(X)

				if all([x.is_integer() for X in data for x in X]):
					X = X[0:numTest,:].astype(dtype=np.int32)
				else:
					X = X[0:numTest,:].astype(dtype=np.float32)

				Y = Y[0:numTest]

			clf = joblib.load(basepath + "/text/" + name + ".pkl")
			print("\tComputing target accuracy")
			YPredicted_ = loadedForest.predict_batch(X)
			YPredictedSK = clf.predict(X)
			# print(clf.classes_)
			# print(YPredictedSK)

			targetAcc = sum(YPredicted_ == Y)
			#print("\tAccuracy MY:%s" % accuracy_score(Y, YPredicted_))
			print("\tWeighted Majority Vote: %s" % sum(YPredictedSK == Y))
			print("\tStandard Majority Vote: %s" % sum(YPredicted_ == Y))
			#print("\tAccuracy SK:%s" % accuracy_score(Y, YPredictedSK))
			#print("\ttargetAcc SK: %s" % sum(YPredictedSK == Y))

			featureType = getFeatureType(X)
			dim = len(X[0])

			Makefile = """COMPILER = {compiler}
FLAGS = -std=c++11 -Wall -O3 -funroll-loops -ftree-vectorize

all:
"""
			print("\tGenerating If-Trees")
			converter = ForestConverter(StandardIFTreeConverter(dim, "StandardIfTree", featureType))
			generateClassifier(cppPath + "/", targetAcc, dim, numTest, converter, "StandardIfTree", featureType, loadedForest, "../../../test.csv", reps)
			Makefile += "\t$(COMPILER) $(FLAGS) StandardIfTree.h StandardIfTree.cpp testStandardIfTree.cpp -o testStandardIfTree" + "\n"

			for s in budgetSizes:
				print("\tIf-Tree for budget", s)

				converter = ForestConverter(OptimizedIFTreeConverter(dim, "OptimizedPathIfTree_" + str(s), featureType, target, "path", s))
				generateClassifier(cppPath + "/", targetAcc, dim, numTest, converter, "OptimizedPathIfTree_"+ str(s), featureType, loadedForest, "../../../test.csv", reps)
				Makefile += "\t$(COMPILER) $(FLAGS) OptimizedPathIfTree_" + str(s)+".h" + " OptimizedPathIfTree_" + str(s)+".cpp testOptimizedPathIfTree_" + str(s)+".cpp -o testOptimizedPathIfTree_" + str(s) + "\n"

				# converter = ForestConverter(OptimizedIFTreeConverter(dim, "OptimizedNodeIfTree_" + str(s), featureType, target, "node", s))
				# generateClassifier(cppPath + "/", targetAcc, dim, numTest, converter, "OptimizedNodeIfTree_" + str(s), featureType, loadedForest, "../../../test.csv", reps)
				# Makefile += "\t$(COMPILER) $(FLAGS) OptimizedNodeIfTree_" + str(s)+".h" + " OptimizedNodeIfTree_" + str(s)+".cpp testOptimizedNodeIfTree_" + str(s)+".cpp -o testOptimizedNodeIfTree_" + str(s) + "\n"

				# converter = ForestConverter(OptimizedIFTreeConverter(dim, "OptimizedSwapIfTree_" + str(s), featureType, target, "swap", s))
				# generateClassifier(cppPath + "/", targetAcc, dim, numTest, converter, "OptimizedSwapIfTree_" + str(s), featureType, loadedForest, "../../../test.csv", reps)
				# Makefile += "\t$(COMPILER) $(FLAGS) OptimizedSwapIfTree_" + str(s)+".h" + " OptimizedSwapIfTree_" + str(s)+".cpp testOptimizedSwapIfTree_" + str(s)+".cpp -o testOptimizedSwapIfTree_" + str(s) + "\n"

			print("\tGenerating NativeTrees")

			converter = ForestConverter(NaiveNativeTreeConverter(dim, "NaiveNativeTree", featureType))
			generateClassifier(cppPath + "/", targetAcc, dim, numTest, converter, "NaiveNativeTree", featureType, loadedForest, "../../../test.csv", reps)
			Makefile += "\t$(COMPILER) $(FLAGS) NaiveNativeTree.h NaiveNativeTree.cpp testNaiveNativeTree.cpp -o testNaiveNativeTree\n"

			converter = ForestConverter(StandardNativeTreeConverter(dim, "StandardNativeTree", featureType))
			generateClassifier(cppPath + "/", targetAcc, dim, numTest, converter, "StandardNativeTree", featureType, loadedForest, "../../../test.csv", reps)
			Makefile += "\t$(COMPILER) $(FLAGS) StandardNativeTree.h StandardNativeTree.cpp testStandardNativeTree.cpp -o testStandardNativeTree\n"

			for s in setSizes:
				print("\tNative for set-size", s)

				converter = ForestConverter(OptimizedNativeTreeConverter(dim, "OptimizedNativeTree_" + str(s), featureType, s))
				generateClassifier(cppPath + "/", targetAcc, dim, numTest, converter, "OptimizedNativeTree_" + str(s), featureType, loadedForest, "../../../test.csv", reps)
				Makefile += "\t$(COMPILER) $(FLAGS) OptimizedNativeTree_" + str(s)+".h" + " OptimizedNativeTree_" + str(s)+".cpp testOptimizedNativeTree_" + str(s)+".cpp -o testOptimizedNativeTree_" + str(s) + "\n"

				# print("\tOptimizedNativeForest for set-size", s)

				# converter = OptimizedNativeForestConverter(OptimizedNativeTreeConverterForest(dim, "OptimizedNativeForest_" + str(s), featureType, s))
				# generateClassifier(cppPath + "/", targetAcc, dim, numTest, converter, "OptimizedNativeForest_" + str(s), featureType, loadedForest, "../../../test.csv", reps)
				# Makefile += "\t$(COMPILER) $(FLAGS) OptimizedNativeForest_" + str(s)+".h" + " OptimizedNativeForest_" + str(s)+".cpp testOptimizedNativeForest_" + str(s)+".cpp -o testOptimizedNativeForest_" + str(s) + "\n"
			# print("\tGenerating MixTrees")
			#converter = ForestConverter(MixConverter(dim, "MixTree", featureType, target))
			#generateClassifier(cppPath + "/", targetAcc, X,Y, converter, "MixTree", featureType, loadedForest, "../../../test.csv", reps)

			if target == "intel":
				compiler = "g++"
			elif target == "ppc":
								compiler = "powerpc-fsl-linux-g++ -m32 -mhard-float -mcpu=e6500 --sysroot=/opt/fsl-qoriq/2.0/sysroots/ppce6500-fsl-linux --static"
			else:
				compiler = "arm-linux-gnueabihf-g++"

			Makefile = Makefile.replace("{compiler}", compiler)

			with open(cppPath + "/" + "Makefile",'w') as code_file:
				code_file.write(Makefile)

		print("")

if __name__ == "__main__":
   main(sys.argv[1:])
