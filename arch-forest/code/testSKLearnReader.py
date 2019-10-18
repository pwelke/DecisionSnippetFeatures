#!/usr/bin/env python3

import sys
import numpy as np
import pprint, json

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

import Tree
import Forest

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

def testModel(XTrain,YTrain,XTest,YTest,m):
	m.fit(XTrain,YTrain)

	mymodel1 = Forest.Forest()
	mymodel2 = Forest.Forest()

	mymodel1.fromSKLearn(m)
	with open("tmp.json",'w') as outFile:
		outFile.write(mymodel1.str())
	joblib.dump(m, "tmp.pkl")

	mymodel2.fromJSON("tmp.json")
	skmodel2 = joblib.load("tmp.pkl")

	mymodelCnt1 = 0
	mymodelCnt2 = 0
	skModelCnt1 = 0
	skModelCnt2 = 0

	for (x,y) in zip(XTest,YTest):
		skpred1 = m.predict(x.reshape(1, -1))[0]
		skpred2 = skmodel2.predict(x.reshape(1, -1))[0]
			
		mypred1 = mymodel1.predict(x)
		mypred1 = mypred1.argmax()
		mypred2 = mymodel2.predict(x)
		mypred2 = mypred2.argmax()

		if (skpred1 != mypred1):
			print(m.predict_proba(x.reshape(1, -1)))
			print(m.predict(x.reshape(1, -1)))
			print(mymodel1.predict(x))
			print(mymodel1.pstr())

			return False
		
		mymodelCnt1 += (mypred1 == y)
		mymodelCnt2 += (mypred2 == y)
		skModelCnt1 += (skpred1 == y)
		skModelCnt2 += (skpred2 == y)

	# YPredictedSK = clf.predict(XTest)
	# targetAcc = sum(YPredictedSK == YTest)
	# print("targetAcc:",targetAcc)
	print("m1:", mymodelCnt1)
	print("m2:", mymodelCnt2)
	print("sk1:", skModelCnt1)
	print("sk2:", skModelCnt2)

	return True

def main(argv):
	#data = load_breast_cancer()
	red = np.genfromtxt("../data/wine-quality/winequality-red.csv", delimiter=';', skip_header=1)
	white = np.genfromtxt("../data/wine-quality/winequality-white.csv", delimiter=';', skip_header=1)
	X = np.vstack((red[:,:-1],white[:,:-1])).astype(dtype=np.float32)
	Y = np.concatenate((red[:,-1], white[:,-1]))
	# NOTE: It seems, that SKLEarn produces an internal mapping from 0-(|Y| - 1) for classification
	# 		For some reason I was not able to extract this mapping from SKLearn ?!?!
	Y = Y-min(Y)

	XTrain,XTest,YTrain,YTest = train_test_split(X, Y, test_size=0.25)
	#X = data.data.astype(dtype=np.float32)
	#Y = data.target

	print("BINARY CLASSIFICATION TEST")
	print("### Decision Tree ###")
	if testModel(XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1), False):
		print("    test passed")

	# print("### Extra Tree ###")
	# if testModel(X,Y,ExtraTreesClassifier(n_estimators=20)):
	# 	print("    test passed")

	# print("### Random Forest ###")
	# if testModel(X,Y,RandomForestClassifier(n_estimators=20)):
	# 	print("    test passed")

	# print("### AdaBoost Classifier ###")
	# if testModel(X,Y,AdaBoostClassifier(n_estimators=20)):
	# 	print("    test passed")

	# print()
	# print()
	# print("MULTICLASS CLASSIFICATION TEST")
	# data = load_iris()
	# X = data.data.astype(dtype=np.float32)
	# Y = data.target
	# print("### Decision Tree ###")
	# if testModel(X,Y,DecisionTreeClassifier(), True):
	# 	print("    test passed")

	# print("### Extra Tree ###")
	# if testModel(X,Y,ExtraTreesClassifier(n_estimators=20)):
	# 	print("    test passed")

	# print("### Random Forest ###")
	# if testModel(X,Y,RandomForestClassifier(n_estimators=20)):
	# 	print("    test passed")

	# print("### AdaBoost Classifier ###")
	# if testModel(X,Y,AdaBoostClassifier(n_estimators=20)):
	# 	print("    test passed")


if __name__ == "__main__":
   main(sys.argv[1:])
