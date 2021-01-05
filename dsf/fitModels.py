# Written by Sebastian Buschjäger 2018
# minor changes by Pascal Welke 2020

import sys
import csv
import numpy as np
import os.path
import json
import timeit

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

sys.path.append('./arch-forest/code/')
import Forest
import Tree


def testModel(roundSplit, XTrain, YTrain, XTest, YTest, model, name, model_dir):
	print("Fitting", name)
	model.fit(XTrain,YTrain)

	print("Testing ", name)
	start = timeit.default_timer()
	YPredicted = model.predict(XTest)
	end = timeit.default_timer()

	print("Total time: " + str(end - start) + " ms")
	print("Throughput: " + str(len(XTest) / (float(end - start)*1000)) + " #elem/ms")

	print("Saving model")
	if (issubclass(type(model), DecisionTreeClassifier)):
		mymodel = Tree.Tree()
	else:
		mymodel = Forest.Forest()

	mymodel.fromSKLearn(model,roundSplit)

	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	with open(os.path.join(model_dir, name + ".json"), 'w') as outFile:
		outFile.write(mymodel.str())

	SKPred = model.predict(XTest)
	MYPred = mymodel.predict_batch(XTest)
	accuracy = accuracy_score(YTest, SKPred)
	print("Accuracy:", accuracy)

	# This can now happen because of classical majority vote
	# for (skpred, mypred) in zip(SKPred,MYPred):
	# 	if (skpred != mypred):
	# 		print("Prediction mismatch!!!")
	# 		print(skpred, " vs ", mypred)

	print("Saving model to PKL on disk")
	joblib.dump(model, os.path.join(model_dir, name + ".pkl"))
	
	print("*** Summary ***")
	print("#Examples\t #Features\t Accuracy\t Avg.Tree Height")
	print(str(len(XTest)) + "\t" + str(len(XTest[0])) + "\t" + str(accuracy) + "\t" + str(mymodel.getAvgDepth()))
	print()


def fitModels(roundSplit, XTrain, YTrain, XTest=None, YTest=None, createTest=False, model_dir='text', 
              types=['RF', 'ET', 'DT'], 
			  forest_depths=[5, 10, 15, 20],
              forest_size=25):
	''' Fit a bunch of forest models to the given train data and write the resulting models to disc.
	Possible forest types are: 
	- DT (decision tree)
	- ET (extra trees)
	- RF (random forest)
	- AB (adaboost) '''
	if XTest is None or YTest is None:
		XTrain,XTest,YTrain,YTest = train_test_split(XTrain, YTrain, test_size=0.25)
		createTest = True

	if createTest:
		with open("test.csv", 'w') as outFile:
			for x,y in zip(XTest, YTest):
				line = str(y)
				for xi in x:
					line += "," + str(xi)

				outFile.write(line + "\n")

	if 'DT' in types:
		for depth in forest_depths:
			testModel(roundSplit, XTrain, YTrain, XTest, YTest, RandomForestClassifier(n_estimators=1, n_jobs=8, max_depth=depth), f"DT_{depth}", model_dir)

	if 'ET' in types:
		for depth in forest_depths:
			testModel(roundSplit, XTrain, YTrain, XTest, YTest, ExtraTreesClassifier(n_estimators=forest_size, n_jobs=8, max_depth=depth), f"ET_{depth}", model_dir)

	if 'RF' in types:
		for depth in forest_depths:
			testModel(roundSplit, XTrain, YTrain, XTest, YTest, RandomForestClassifier(n_estimators=forest_size, n_jobs=8, max_depth=depth), f"RF_{depth}", model_dir)

