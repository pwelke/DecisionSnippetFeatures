#!/usr/bin/env python3

import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.tree import _tree

def testModel(X,Y,m):
	m.fit(X,Y)

	for (x,y) in zip(X,Y):
		skpred = m.predict(x.reshape(1, -1))[0]

		n_nodes = m.estimators_[0].tree_.node_count
		children_left = m.estimators_[0].tree_.children_left
		children_right = m.estimators_[0].tree_.children_right
		feature = m.estimators_[0].tree_.feature
		threshold = m.estimators_[0].tree_.threshold
		
		dpath = m.estimators_[0].decision_path(x.reshape(1, -1)).toarray()[0]
		skpath = np.where(dpath == 1)[0]

		#print("skpath:", skpath) # [0][0:5]

		mypred = None
		nodeid = 0
		mypath = []
		fvalues = []
		thresholds = []

		while(True):
			if children_left[nodeid] == _tree.TREE_LEAF and children_right[nodeid] == _tree.TREE_LEAF:
				#prediction = tree.value[curNode][0, :]
				break; # prediction found
			else:
				mypath.append(nodeid)
				fvalues.append(feature[nodeid])
				thresholds.append(threshold[nodeid])

				if (x[feature[nodeid]] <= threshold[nodeid]):
					nodeid = children_left[nodeid]
				else:
					nodeid = children_right[nodeid]
		
		#print("mypath:",path)
		for i,j in zip(skpath,mypath):
			if i != j:
				print("Mismatch in paths detected!")
				print("skpath:", skpath)
				print("mypath:", np.array(mypath))
				print("xdtype:", x.dtype)
				for (nid,fval,th) in zip(mypath,fvalues,thresholds):
					print(nid, ":", x[fval] , " <= " , th , "=", x[fval] <= th)
					
				#print("fvalues:", np.array(fvalues))
				#print("thresholds:", np.array(thresholds))

				print("The binary tree structure has %s nodes and has "
						"the following tree structure:" % n_nodes)
				
				# The tree structure can be traversed to compute various properties such
				# as the depth of each node and whether or not it is a leaf.
				node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
				is_leaves = np.zeros(shape=n_nodes, dtype=bool)
				stack = [(0, -1)]  # seed is the root node id and its parent depth
				while len(stack) > 0:
					node_id, parent_depth = stack.pop()
					node_depth[node_id] = parent_depth + 1

					# If we have a test node
					if (children_left[node_id] != children_right[node_id]):
						stack.append((children_left[node_id], parent_depth + 1))
						stack.append((children_right[node_id], parent_depth + 1))
					else:
						is_leaves[node_id] = True

				for i in range(n_nodes):
					if is_leaves[i]:
						print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
					else:
						print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
							  "node %s."
							  % (node_depth[i] * "\t",
								 i,
								 children_left[i],
								 feature[i],
								 threshold[i],
								 children_right[i],
								 ))
				return

def main(argv):
	data = load_breast_cancer()

	# IMPORTANT: RandomForestClassifier internally converts the data provided to np.float32
	#	In some rare cases, this may lead to different prediction outcomes since 
	#   data is converted into different types. To prevent this, we load the data as float32
	#	even though the standard is float64. Alternativley, we may use "check_array()" from
	#	validation.py in utils so perform this conversion. This would exactly match the way
	#	sklearn does it in the predict() method
	X = data.data.astype(dtype=np.float32)
	Y = data.target

	print("### Random Forest ###")
	for i in range(0,200):
		testModel(X,Y,RandomForestClassifier(n_estimators=1))

if __name__ == "__main__":
   main(sys.argv[1:])
