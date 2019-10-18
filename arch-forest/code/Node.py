import numpy as np
from sklearn.tree import _tree

class Node:
	def __init__(self):
		# The ID of this node. Makes addressing sometimes easier 
		self.id = None
		
		# The total number of samples seen at this node 
		self.numSamples = None
		
		# The probability to follow the left child
		self.probLeft = None
		
		# The probability to follow the right child
		self.probRight = None

		# An array of predictions, where each entry represents the class weight / probability
		# The specific meaning of each entry depends on the specific method (AdaBoost vs RandomForest)
		self.prediction = None

		# The 'way' of comparison. Usually this is "<=". However, for non-numeric 
		# features, one sometimes need equals comparisons "==". This is currently not 
		# implemented. 
		self.isCategorical = None

		# The index of the feature to compare against
		self.feature = None

		# The threashold the features gets compared against
		self.split = None

		# The right child of this node inside the tree
		self.rightChild = None

		# The left child of this node inside the tree
		self.leftChild = None

		# The probability of this node accumulated from the probabilities of previous
        # edges on the same path.
        # Note: This field is only used after calling getProbAllPaths onc
		self.pathProb = None

	# TODO: THESE CHANGES ARE CURRENTLY JUST NEEDED BY Tree.py FOR MIXTURE IMPLEMENTATION
	# Unfortunately, as the standard library provides min-heap, I invert the object comparison
	def __lt__(self, other):
		return self.pathProb > other.pathProb

	def __eq__(self, other):
		return self.pathProb == other.pathProb
	
	def __str__(self):
		return str(self.id)

	def fromSKLearn(self, tree, curNode, roundSplit = False, skType = "RandomForest", weight = 1.0):
		"""Generate a node from a sci-kit tree
		
		Args:
		    tree: The (internal) sci-kit tree object 
		    curNode: The index of the current node
		
		Returns:
		    Node: An node representing the given (internal) sci-kit node
		"""
		self.numSamples = int(tree.n_node_samples[curNode])
		if tree.children_left[curNode] == _tree.TREE_LEAF and tree.children_right[curNode] == _tree.TREE_LEAF:
			# Get array of prediction probabilities for each class
			proba = tree.value[curNode][0, :]  
			#print(proba/sum(proba))
			#print(proba)
			#proba = proba / sum(proba)  #TODO??

			nClasses = len(proba)

			if skType == "SAMME.R":
				proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
				log_proba = np.log(proba)

				proba = (nClasses - 1) * (log_proba - (1. / nClasses) * sum(log_proba))
			elif skType == "RandomForest":
				#proba = proba / tree.weighted_n_node_samples[curNode]
				proba = proba / sum(proba)#tree.weighted_n_node_samples[curNode]

			self.prediction = proba * weight
		else:
			self.feature = tree.feature[curNode]

			if (roundSplit):
				self.split = int(tree.threshold[curNode])
			else:
				self.split = tree.threshold[curNode]

			self.isCategorical = False # Note: So far, sklearn does only support numrical features, see https://github.com/scikit-learn/scikit-learn/pull/4899
			samplesLeft = float(tree.n_node_samples[tree.children_left[curNode]])
			samplesRight = float(tree.n_node_samples[tree.children_right[curNode]])
			self.probLeft = samplesLeft / self.numSamples
			self.probRight = samplesRight / self.numSamples

	def fromJSON(self, json):
		self.id = json["id"]
		self.numSamples = int(json["numSamples"])

		if "prediction" in json:
			self.prediction = json["prediction"]
		else:
			self.probLeft = float(json["probLeft"])
			self.probRight = float(json["probRight"])
			self.isCategorical = (json["isCategorical"] == "True")
			self.feature = int(json["feature"])
			self.split = json["split"]
			self.rightChild = json["rightChild"]["id"]
			self.leftChild = json["leftChild"]["id"]

	def fromNode(self, node):
		""" Simple copy constructor
		
		Args:
		    node (Node): The node to be copied
		
		Returns:
		    Node: A copy of the given node
		"""
		self.id = node.id
		self.numSamples = node.numSamples
		self.probLeft = node.probLeft
		self.probRight = node.probRight
		self.prediction = node.prediction
		self.isCategorical = node.isCategorical
		self.feature = node.feature
		self.split = node.split
		self.rightChild = node.rightChild
		self.leftChild = node.leftChild

	def str(self, leftChilds = "", rightChilds = ""):
		""" Returns a JSON-String representation of the node
		
		Returns:
		    TYPE: The JSON-String representation of the node
		"""
		s = ""

		s = "{"
		s += "\"id\":" + str(self.id) + ","
		s += "\"numSamples\":" + str(self.numSamples) + ","
		
		if self.prediction is not None:
			sp = "["
			for e in self.prediction:
				sp += str(e) + "," 
			sp = sp[:-1] + "]"

			s += "\"prediction\":" + str(sp)
			#s += "\"prediction\":\"" + str([]) + "\""
		else:
			s += "\"probLeft\":" + str(self.probLeft) + ","
			s += "\"probRight\":" + str(self.probRight) + ","
			s += "\"isCategorical\":\"" + str(self.isCategorical) + "\","
			s += "\"feature\":" + str(self.feature) + ","
			s += "\"split\":" + str(self.split) + ","
			s += "\"leftChild\":" + leftChilds + ","
			s += "\"rightChild\": " + rightChilds 
		s += "}"

		return s

	def predict(self,x):
		return np.argmax(self.prediction)
		#return np.array(self.prediction)