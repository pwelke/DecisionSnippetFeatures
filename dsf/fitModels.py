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

def fitModels(roundSplit, XTrain, YTrain, XTest=None, YTest=None, createTest=False, model_dir='text'):
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

	# testModel(XTrain,YTrain,XTest,YTest,DecisionTreeClassifier(max_depth=1),"DT_1")
	# testModel(XTrain,YTrain,XTest,YTest,DecisionTreeClassifier(max_depth=5),"DT_5")
	# testModel(XTrain,YTrain,XTest,YTest,DecisionTreeClassifier(max_depth=15),"DT_15")
	# testModel(XTrain,YTrain,XTest,YTest,DecisionTreeClassifier(max_depth=None),"DT_unlimited")

	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=8,max_depth=1),"DT_1", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=8,max_depth=5),"DT_5", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=8,max_depth=10),"DT_10", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=8,max_depth=15),"DT_15", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=8,max_depth=20),"DT_20", model_dir)
	#testModel(XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=4,max_depth=None),"DT_unlimited", model_dir)

	testModel(roundSplit,XTrain,YTrain,XTest,YTest,ExtraTreesClassifier(n_estimators=25,n_jobs=8,max_depth=1),"ET_1", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,ExtraTreesClassifier(n_estimators=25,n_jobs=8,max_depth=5),"ET_5", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,ExtraTreesClassifier(n_estimators=25,n_jobs=8,max_depth=10),"ET_10", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,ExtraTreesClassifier(n_estimators=25,n_jobs=8,max_depth=15),"ET_15", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,ExtraTreesClassifier(n_estimators=25,n_jobs=8,max_depth=20),"ET_20", model_dir)
	#testModel(XTrain,YTrain,XTest,YTest,ExtraTreesClassifier(n_estimators=25,n_jobs=4,max_depth=None),"ET_unlimited", model_dir)

	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=8,max_depth=1),"RF_1", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=8,max_depth=5),"RF_5", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=8,max_depth=10),"RF_10", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=8,max_depth=15),"RF_15", model_dir)
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=8,max_depth=20),"RF_20", model_dir)
	#testModel(XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=4,max_depth=None),"RF_unlimited", model_dir)

	#testModel(XTrain,YTrain,XTest,YTest,AdaBoostClassifier(n_estimators=50,base_estimator=DecisionTreeClassifier(max_depth=1)),"AB_1", model_dir)
	#testModel(XTrain,YTrain,XTest,YTest,AdaBoostClassifier(n_estimators=50,base_estimator=DecisionTreeClassifier(max_depth=3)),"AB_3", model_dir)
	#testModel(XTrain,YTrain,XTest,YTest,AdaBoostClassifier(n_estimators=50,base_estimator=DecisionTreeClassifier(max_depth=5)),"AB_5", model_dir)
