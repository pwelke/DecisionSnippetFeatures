import json
from functools import reduce

import numpy as np

from sklearn.tree import _tree

import Node

class Tree():
	def __init__(self):
		# For simpler computations of statistics, we will also save all nodes in a
		# dictionary where (key = nodeID, value = actuale node)
		self.nodes = {}

		# Pointer to the root node of this tree
		self.head = None
		self.numClasses = None

	def getNumClasses(self):
		return self.numClasses

	def fromTree(self, nodes, head):
		self.nodes = nodes
		self.head = head

	def fromJSON(self, json, first = True):
		node = Node.Node()
		node.fromJSON(json)
		self.nodes[node.id] = node
		if node.prediction is None:
			node.rightChild = self.fromJSON(json["rightChild"], False)
			node.leftChild = self.fromJSON(json["leftChild"], False)
		elif self.numClasses is None:
			self.numClasses = len(node.prediction)

		if first:
			self.head = node

		return node

	def fromSKLearn(self, tree, roundSplit = False, skType = "RandomForest", weight = 1.0):
		self.head = self._fromSKLearn(tree.tree_, 0, roundSplit, skType, weight)

	def _fromSKLearn(self, tree, curNode, roundSplit = False, skType = "RandomForest", weight = 1.0):
		""" Loads a tree from sci-kit internal data structure into this object
		
		Args:
		    tree (TYPE): The sci-kit tree
		    curNode (int, optional): The current node index (default = 0 ==> root node of the tree)
		
		Returns:
		    TYPE: The root node of the extracted tree structure
		"""
		node = Node.Node()
		node.fromSKLearn(tree, curNode, roundSplit, skType, weight)
		node.id = len(self.nodes)
		self.nodes[node.id] = node

		if node.prediction is None:
			leftChild = tree.children_left[curNode]
			rightChild = tree.children_right[curNode]
			
			node.leftChild = self._fromSKLearn(tree, leftChild, roundSplit, skType, weight)
			node.rightChild = self._fromSKLearn(tree, rightChild, roundSplit, skType, weight)
		elif self.numClasses is None:
			self.numClasses = len(node.prediction)

		return node

	def str(self, head = None):
		if head is None:
			head = self.head

		if head.prediction is not None:
			return head.str()
		else:
			leftChilds = self.str(head.leftChild)
			rightChilds = self.str(head.rightChild)
			s = head.str(leftChilds, rightChilds)
			return s
	
	def pstr(self):
		parsed = json.loads(self.str())
		return json.dumps(parsed, indent=4)

	## SOME STATISTICS FUNCTIONS ##
	def getSubTree(self, minProb, maxNumNodes):
		allSubPaths = self.getAllPaths()
		allSubPaths.sort(key = lambda x : len(x), reverse=True)
		paths = []
		curProb = 1.0
		curSize = 0

		added = True
		while(added):
			added = False
			for p in allSubPaths:
				prob = np.prod([n[1] for n in p])

				if curProb + prob > minProb and curSize + len(p) < maxNumNodes:
					paths.append(p)
					curSize += len(p)
					curProb += prob

					added = True

					# Does this work during iteration?!
					break
			
			if (added):
				allSubPaths.remove(paths[-1])

		return paths,curProb,curSize

	def getProbAllPaths(self, node = None, curPath = None, allPaths = None, pathNodes = None, pathLabels = None):
		if node is None:
			node = self.head

		if curPath is None:
			curPath = []

		if allPaths is None:
			allPaths = []

		if pathNodes is None:
			pathNodes = []

		if pathLabels is None:
			pathLabels = []

		if node.prediction is not None:
			allPaths.append(curPath)
			pathLabels.append(pathNodes)
			curProb = reduce(lambda x, y: x*y, curPath)
			node.pathProb = curProb
			#print("Leaf nodes "+str(node.id)+" : "+str(curProb))
		else:
			if len(pathNodes) == 0:
				curPath.append(1)

			pathNodes.append(node.id)
			# try:
			# 	pathNodes.index(node.id)
			# 	#this node is root
			# except ValueError:
			# 	pathNodes.append(node.id)
			# 	curPath.append(1)

			curProb = reduce(lambda x, y: x*y, curPath)
			node.pathProb = curProb
			#print("Root or Split nodes "+str(node.id)+ " : " +str(curProb))
			self.getProbAllPaths(node.leftChild, curPath + [node.probLeft], allPaths, pathNodes + [node.leftChild.id], pathLabels)
			self.getProbAllPaths(node.rightChild, curPath + [node.probRight], allPaths, pathNodes + [node.rightChild.id], pathLabels)

		return allPaths, pathLabels

	# def getNumNodes(self):
	# 	return len(self.nodes)

	# def getMaxDepth(self):
	# 	paths = self.getAllPaths()
	# 	return max([len(p) for p in paths])

	def getAvgDepth(self):
		paths = self.getAllPaths()
		return sum([len(p) for p in paths]) / len(paths)
	
	def getAllLeafPaths(self, node = None, curPath = None, allPaths = None):
		# NOTE: FOR SOME REASON allPaths = [] DOES NOT CORRECTLY RESET
		# 		THE allPaths VARIABLE. THUS WE USE NONE HERE 
		if allPaths is None:
			allPaths = []

		if curPath is None:
			curPath = []

		if node is None:
			node = self.head

		if node.prediction is not None:
			curPath.append((node.id,1))
			allPaths.append(curPath)
		else:
			self.getAllLeafPaths(node.leftChild, curPath + [(node.id,node.probLeft)], allPaths)
			self.getAllLeafPaths(node.rightChild, curPath + [(node.id,node.probRight)], allPaths)
		
		return allPaths

	# This returns all sub-paths starting with the root node
	def getAllPaths(self, node = None, curPath = None, allPaths = None):
		# NOTE: FOR SOME REASON allPaths = [] DOES NOT CORRECTLY RESET
		# 		THE allPaths VARIABLE. THUS WE USE NONE HERE 
		if allPaths is None:
			allPaths = []

		if curPath is None:
			curPath = []

		if node is None:
			node = self.head

		if node.prediction is not None:
			curPath.append((node.id,1))
			allPaths.append(curPath)
		else:
			# We wish to include all sub-paths, not only complete paths from root to leaf node
			if len(curPath) > 0:
				allPaths.append(curPath)

			self.getAllPaths(node.leftChild, curPath + [(node.id,node.probLeft)], allPaths)
			self.getAllPaths(node.rightChild, curPath + [(node.id,node.probRight)], allPaths)
		
		return allPaths

	def getNumNodes(self):
		return len(self.nodes)

	def predict(self,x):
		curNode = self.head

		while(curNode.prediction is None):

			if (x[curNode.feature] <= curNode.split): 
				curNode = curNode.leftChild
			else:
				curNode = curNode.rightChild


		return curNode.predict(x)

	def predict_batch(self,X):
		YPred = []
		for x in X:
			YPred.append(self.predict(x).argmax())
			
		return YPred
	# def getMaxProb(self, top_n = 1):
	# 	paths = self.getAllPaths()
	# 	probs = [reduce(lambda x, y: x*y, path) for path in paths]
	# 	probs.sort(reverse=True)

	# 	return probs[0:top_n]

	# def getAvgProb(self):
	# 	paths = self.getAllPaths()
	# 	return sum( [reduce(lambda x, y: x*y, path) for path in paths] ) / len(paths)