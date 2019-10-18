#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np
import os.path
import json
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def readFile(path):
	f = open(path, 'r')
	X = []
	Y = []
	for row in f:
		entries = row.strip().split(",")
		x = [float(e) for e in entries[1:]]
		y = float(entries[0])
		X.append(x)
		Y.append(y)

	return np.array(X), np.array(Y)

def main(argv):
	if len(argv)<1:
		print("Please give a sub-folder / dataset to be used")
		return
	else:
		basepath = argv[0].strip("/")

	#print("Reading test file")
	XTest,YTest = readFile(basepath + "/test.csv")
	
	# Preparse fair testing data
	maxn = min(len(XTest),10000)
	indices = np.random.choice(len(XTest),maxn)	
	XTest_ = []	
	YTest_ = []
	
	for i in indices:
		XTest_.append(np.expand_dims(XTest[i],axis=0))
		YTest_.append(YTest[i])
	
	for f in sorted(os.listdir(basepath + "/text/")):
		if f.endswith(".pkl"): 
			#print("Loading model", f)
			clf = joblib.load(basepath + "/text/" + f)
			clf.n_jobs = 1
			acc = 0
			# Burn in phase
			for i in range(2):
				for x,y in zip(XTest_,YTest_):
					ypred = clf.predict(x)
					if (ypred == y):
                	                        acc += 1
			
			#print("Accuracy:%s" % accuracy_score(YTest, YPredicted))	
				
			# Actual measurement
			runtimes = []
			acc = 0
			for i in range(5):
				start = timeit.default_timer()
				for x,y in zip(XTest_,YTest_):
					ypred = clf.predict(x)
					if (ypred == y):
						acc += 1
				
				# NOTE: Usually, one would call this method on all data-points. This enables SKLearn / python just in time compilation / the C-backend to utilize parallelsim due to multiple instances available. We do not want to compare against this, because in deployment we usally have one new observation after another which should be classified as fast as possible.
				# NOTE 2: It is probably possible to buffer multiple instances in real applications. However, this sort of optimization is different from what we are looking into here
				# Note 3: Interestringly, in most cases we are still factor 2-4 faster than SKLearn. Only when datasets get bigger, such as FACT, SKLearn can take full adavantage of having all datapoints available at a time.
				#YPredicted = clf.predict(XTest)
				end = timeit.default_timer()
				runtimes.append((end-start)/maxn*1000)

			print(basepath+"/text/"+f,",",np.mean(runtimes),",",np.var(runtimes))			

if __name__ == "__main__":
   main(sys.argv[1:])
