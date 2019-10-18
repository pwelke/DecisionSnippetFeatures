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


def plotType(filename,targettype):
	results = {}
	f = open(filename, 'r')
	header = next(f)
	for row in f:
		entries = row.replace("\n","").split(",")
		exp = entries[1].replace("test","")
		treetype = entries[2].split("_")[0]

		if (treetype == targettype):
			treedepth = int(entries[2].split("_")[1])

			if not exp in results:
				results[exp] = {}

			if not treedepth in results[exp]:
				results[exp][treedepth] = []

			results[exp][treedepth].append(float(entries[3]))

	baseexp = "NaiveNativeTree"
	speedup = {}

	for exp in results:
		if exp not in speedup:
			speedup[exp] = {}

		for depth in results[exp]:
			if depth not in speedup[exp]:
				speedup[exp][depth] = []

			for val,ref in zip(results[exp][depth], results[baseexp][depth]):
				speedup[exp][depth].append(ref/val)	

	tikz = """
		\\documentclass[tikz,border=5pt]{standalone}
		\\usepackage{pgfplots}
		\\pgfplotsset{compat=newest}
		\\begin{document}
		\\begin{tikzpicture}
		\\begin{axis}[legend style={nodes={scale=0.5, transform shape}}]
	"""

	for exp in results:
		if (exp != baseexp):
			tikz += "\\addplot+[error bars/.cd,y dir=both,y explicit]\n"
			tikz += "coordinates {\n"
			for depth in sorted(speedup[exp]):
				minVal = min(speedup[exp][depth])
				maxVal = max(speedup[exp][depth])
				avgVal = np.mean(speedup[exp][depth])
				#avgVal = sum(speedup[exp][depth])/len(speedup[exp][depth])

				#minEps = avgVal - minVal
				#maxEps = maxVal - avgVal
				minEps = np.std(speedup[exp][depth], axis=0)
				maxEps = minEps

				tikz += "({depth},{avgVal}) +- ({max},{min})\n".replace("{depth}", str(depth)).replace("{avgVal}", str(avgVal)).replace("{min}",str(minEps)).replace("{max}", str(maxEps))
			tikz += "};\n"
			tikz += "\\addlegendentry{{exp}}\n\n".replace("{exp}",exp.replace("_","\_"))

	tikz += """
		\\end{axis}
		\\end{tikzpicture}
		\\end{document}
	"""
	return tikz

def main(argv):
	if len(argv) < 1:
		print("Please provide a filename for plotting")
		return
	else:
		filename = argv[0]

	if len(argv) < 2:
		print("Please provide a folder where the plot files should be stored")
		return
	else:
		outpath = argv[1]

	# "path,filename,depth,mean,variance,min,max,size"
	treetypes = ["RF", "DT", "ET"]
	for treetype in treetypes:
		tikz = plotType(filename,treetype)
		with open(outpath + "/" + filename.split(".")[0] + "_" + treetype + ".tex",'w') as plotFile:
			plotFile.write(tikz)

	#print(tikz)
	#print(results)
	#print(speedup)

if __name__ == "__main__":
   main(sys.argv[1:])
