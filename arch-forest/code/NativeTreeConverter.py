from ForestConverter import TreeConverter
import numpy as np
import heapq

class NativeTreeConverter(TreeConverter):
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

    def getArrayLenType(self, arrLen):
            arrayLenBit = int(np.log2(arrLen)) + 1
            if arrayLenBit <= 8:
                    arrayLenDataType = "unsigned char"
            elif arrayLenBit <= 16:
                    arrayLenDataType = "unsigned short"
            else:
                    arrayLenDataType = "unsigned int"
            return arrayLenDataType

    def getImplementation(self, head, treeID):
        raise NotImplementedError("This function should not be called directly, but only by a sub-class")

    def getHeader(self, splitType, treeID, arrLen, numClasses):
            dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

            if dimBit <= 8:
                    dimDataType = "unsigned char"
            elif dimBit <= 16:
                    dimDataType = "unsigned short"
            else:
                    dimDataType = "unsigned int"

            featureType = self.getFeatureType()
            if (numClasses == 2):
                headerCode = """struct {namespace}_Node{treeID} {
                        //bool isLeaf;
                        //unsigned int prediction;
                        {dimDataType} feature;
                        {splitType} split;
                        {arrayLenDataType} leftChild;
                        {arrayLenDataType} rightChild;
                        unsigned char indicator;

                };\n""".replace("{namespace}", self.namespace) \
                           .replace("{treeID}", str(treeID)) \
                           .replace("{splitType}",splitType) \
                           .replace("{dimDataType}",dimDataType) \
                           .replace("{arrayLenDataType}", self.getArrayLenType(arrLen))
            else:
                headerCode = """struct {namespace}_Node{treeID} {
                           //bool isLeaf;
                            {dimDataType} feature;
                            {splitType} split;
                            {arrayLenDataType} leftChild;
                            {arrayLenDataType} rightChild;
                            unsigned char indicator;
                };\n""".replace("{namespace}", self.namespace) \
                       .replace("{treeID}", str(treeID)) \
                       .replace("{splitType}",splitType) \
                       .replace("{dimDataType}",dimDataType) \
                       .replace("{arrayLenDataType}", self.getArrayLenType(arrLen))

            headerCode += "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                            .replace("{treeID}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType)
            return headerCode

    def getCode(self, tree, treeID, numClasses):
            # kh.chen
            # Note: this function has to be called once to traverse the tree to calculate the probabilities.
            tree.getProbAllPaths()
            cppCode, arrLen = self.getImplementation(tree.head, treeID)

            if self.containsFloat(tree):
                splitDataType = "float"
            else:
                lower, upper = self.getSplitRange(tree)

                bitUsed = 0
                if lower > 0:
                    prefix = "unsigned"
                    maxVal = upper
                else:
                    prefix = ""
                    bitUsed = 1
                    maxVal = max(-lower, upper)

                splitBit = int(np.log2(maxVal) + 1 if maxVal != 0 else 1)

                if splitBit <= (8-bitUsed):
                    splitDataType = prefix + " char"
                elif splitBit <= (16-bitUsed):
                    splitDataType = prefix + " short"
                else:
                    splitDataType = prefix + " int"
            headerCode = self.getHeader(splitDataType, treeID, arrLen, numClasses)

            return headerCode, cppCode

class NaiveNativeTreeConverter(NativeTreeConverter):
    def __init__(self, dim, namespace, featureType):
            super().__init__(dim, namespace, featureType)

    def getHeader(self, splitType, treeID, arrLen, numClasses):
            dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

            if dimBit <= 8:
                    dimDataType = "unsigned char"
            elif dimBit <= 16:
                    dimDataType = "unsigned short"
            else:
                    dimDataType = "unsigned int"

            featureType = self.getFeatureType()
            headerCode = """struct {namespace}_Node{treeID} {
                    bool isLeaf;
                    unsigned int prediction;
                    {dimDataType} feature;
                    {splitType} split;
                    {arrayLenDataType} leftChild;
                    {arrayLenDataType} rightChild;
            };\n""".replace("{namespace}", self.namespace) \
                       .replace("{treeID}", str(treeID)) \
                       .replace("{arrayLenDataType}", self.getArrayLenType(arrLen)) \
                       .replace("{splitType}",splitType) \
                       .replace("{dimDataType}",dimDataType)

            headerCode += "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                            .replace("{treeID}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType)

            return headerCode

    def getImplementation(self, head, treeID):
            arrayStructs = []
            nextIndexInArray = 1

            # BFS part
            nodes = [head]
            while len(nodes) > 0:
                node = nodes.pop(0)
                entry = []

                if node.prediction is not None:
                    entry.append(1)
                    entry.append(int(np.argmax(node.prediction)))
                    #entry.append(int(node.prediction.at(np.argmax(node.prediction)))
                    #entry.append(node.id)
                    entry.append(0)
                    entry.append(0)
                    entry.append(0)
                    entry.append(0)
                else:
                    entry.append(0)
                    entry.append(0) # COnstant prediction
                    entry.append(node.feature)
                    entry.append(node.split)
                    entry.append(nextIndexInArray)
                    nextIndexInArray += 1
                    entry.append(nextIndexInArray)
                    nextIndexInArray += 1

                    nodes.append(node.leftChild)
                    nodes.append(node.rightChild)

                arrayStructs.append(entry)

            featureType = self.getFeatureType()
            arrLen = len(arrayStructs)
            # kh.chen
            #print("Get ArrayLenType")
            #print(self.getArrayLenType(len(arrayStructs)))

            cppCode = "{namespace}_Node{treeID} const tree{treeID}[{N}] = {" \
                    .replace("{treeID}", str(treeID)) \
                    .replace("{N}", str(len(arrayStructs))) \
                    .replace("{namespace}", self.namespace)

            for e in arrayStructs:
                    cppCode += "{"
                    for val in e:
                            cppCode += str(val) + ","
                    cppCode = cppCode[:-1] + "},"
            cppCode = cppCode[:-1] + "};"

            cppCode += """
                    inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){
                            {arrayLenDataType} i = 0;
                            while(!tree{treeID}[i].isLeaf) {
                                    if (pX[tree{treeID}[i].feature] <= tree{treeID}[i].split){
                                        i = tree{treeID}[i].leftChild;
                                    } else {
                                        i = tree{treeID}[i].rightChild;
                                    }
                            }
                            return tree{treeID}[i].prediction;
                    }
            """.replace("{treeID}", str(treeID)) \
               .replace("{dim}", str(self.dim)) \
               .replace("{namespace}", self.namespace) \
               .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
               .replace("{feature_t}", featureType)

            return cppCode, arrLen

class StandardNativeTreeConverter(NativeTreeConverter):
    def __init__(self, dim, namespace, featureType):
            super().__init__(dim, namespace, featureType)

    def getImplementation(self, head, treeID):
            arrayStructs = []
            nextIndexInArray = 1

            # BFS part
            nodes = [head]
            while len(nodes) > 0:
                    node = nodes.pop(0)
                    entry = []

                    if node.prediction is not None:
                        continue
                        # #print("leaf:"+str(node.id))
                        # entry.append(1)
                        # entry.append(int(node.prediction))
                        # #entry.append(node.id)
                        # entry.append(0)
                        # entry.append(0)
                        # entry.append(0)
                        # entry.append(0)
                    else:
                        #entry.append(0)
                        #entry.append(0) # COnstant prediction
                        #entry.append(node.id)
                        entry.append(node.feature)
                        entry.append(node.split)

                        if (node.leftChild.prediction is not None) and (node.rightChild.prediction is not None):
                            indicator = 3
                            # entry.append(int(node.leftChild.prediction))
                            # entry.append(int(node.rightChild.prediction))
                            entry.append(int(np.argmax(node.leftChild.prediction)))
                            entry.append(int(np.argmax(node.rightChild.prediction)))
                        elif (node.leftChild.prediction is None) and (node.rightChild.prediction is not None):
                            indicator = 2
                            entry.append(nextIndexInArray)
                            nextIndexInArray += 1
                            #entry.append(int(node.rightChild.prediction))
                            entry.append(int(np.argmax(node.rightChild.prediction)))
                        elif (node.leftChild.prediction is not None) and (node.rightChild.prediction is  None):
                            indicator = 1
                            #entry.append(int(node.leftChild.prediction))
                            entry.append(int(np.argmax(node.leftChild.prediction)))
                            entry.append(nextIndexInArray)
                            nextIndexInArray += 1
                        else:
                            indicator = 0
                            entry.append(nextIndexInArray)
                            nextIndexInArray += 1
                            entry.append(nextIndexInArray)
                            nextIndexInArray += 1
                        entry.append(indicator)

                        nodes.append(node.leftChild)
                        nodes.append(node.rightChild)

                    arrayStructs.append(entry)

            featureType = self.getFeatureType()
            arrLen = len(arrayStructs)
            # kh.chen
            #print("Get ArrayLenType")
            #print(self.getArrayLenType(len(arrayStructs)))

            cppCode = "{namespace}_Node{treeID} const tree{treeID}[{N}] = {" \
                    .replace("{treeID}", str(treeID)) \
                    .replace("{N}", str(len(arrayStructs))) \
                    .replace("{namespace}", self.namespace)

            for e in arrayStructs:
                    cppCode += "{"
                    for val in e:
                            cppCode += str(val) + ","
                    cppCode = cppCode[:-1] + "},"
            cppCode = cppCode[:-1] + "};"



            cppCode += """
                    inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){
                            {arrayLenDataType} i = 0;

                            while(true) {
                                if (pX[tree{treeID}[i].feature] <= tree{treeID}[i].split){
                                    if (tree{treeID}[i].indicator == 0 || tree{treeID}[i].indicator == 2) {
                                        i = tree{treeID}[i].leftChild;
                                    } else {
                                        return tree{treeID}[i].leftChild;
                                    }
                                } else {
                                    if (tree{treeID}[i].indicator == 0 || tree{treeID}[i].indicator == 1) {
                                        i = tree{treeID}[i].rightChild;
                                    } else {
                                        return tree{treeID}[i].rightChild;
                                    }
                                }
                            }

                            return 0; // Make the compiler happy
                    }
            """.replace("{treeID}", str(treeID)) \
               .replace("{dim}", str(self.dim)) \
               .replace("{namespace}", self.namespace) \
               .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
               .replace("{feature_t}", featureType)
            return cppCode, arrLen

class OptimizedNativeTreeConverter(NativeTreeConverter):
    def __init__(self, dim, namespace, featureType, setSize = 3):
        super().__init__(dim, namespace, featureType)
        self.setSize = setSize

    def getImplementation(self, head, treeID):
        arrayStructs = []
        nextIndexInArray = 1

        # Path-oriented Layout
        head.parent = -1 #for root init
        L = [head]
        heapq.heapify(L)
        while len(L) > 0:
                #the one with the maximum probability will be the next sub-root.
                node = heapq.heappop(L)
                #print("subroot:"+str(node.id))
                cset = []
                while len(cset) != self.setSize: # 32/10
                    # cset.append(node)
                    entry = []

                    if node.prediction is not None:
                        break
                    else:
                        cset.append(node)
                        #print("split:"+str(node.id))
                        #entry.append(0)
                        #entry.append(0) # Constant prediction
                        #entry.append(node.id)
                        entry.append(node.feature)
                        entry.append(node.split)

                        if (node.leftChild.prediction is not None) and (node.rightChild.prediction is not None):
                            indicator = 3
                            # entry.append(int(node.leftChild.prediction))
                            # entry.append(int(node.rightChild.prediction))
                            entry.append(int(np.argmax(node.leftChild.prediction)))
                            entry.append(int(np.argmax(node.rightChild.prediction)))
                        elif (node.leftChild.prediction is None) and (node.rightChild.prediction is not None):
                            indicator = 2
                            entry.append(-1)
                            node.leftChild.parent = nextIndexInArray - 1

                            # entry.append(int(node.rightChild.prediction))
                            entry.append(int(np.argmax(node.rightChild.prediction)))
                        elif (node.leftChild.prediction is not None) and (node.rightChild.prediction is  None):
                            indicator = 1
                            # entry.append(int(node.leftChild.prediction))
                            entry.append(int(np.argmax(node.leftChild.prediction)))
                            entry.append(-1)
                            node.rightChild.parent = nextIndexInArray - 1

                        else:
                            indicator = 0
                            entry.append(-1)
                            node.leftChild.parent = nextIndexInArray - 1
                            entry.append(-1)
                            node.rightChild.parent = nextIndexInArray - 1
                        entry.append(indicator)

                        # node.leftChild.parent = nextIndexInArray - 1
                        # node.rightChild.parent = nextIndexInArray - 1
                        if node.parent != -1:
                            # if this node is not root, it must be assigned with self.side
                            if node.side == 0:
                                if arrayStructs[node.parent][2] == -1:
                                    arrayStructs[node.parent][2] = nextIndexInArray - 1
                                else:
                                    print("BUG in parent.left")
                            else:
                                if arrayStructs[node.parent][3] == -1:
                                    arrayStructs[node.parent][3] = nextIndexInArray - 1
                                else:
                                    print("BUG in parent.right")

                        # the following two fields now are modified by its children.
                        # entry.append(-1)
                        # entry.append(-1)
                        arrayStructs.append(entry)
                        nextIndexInArray += 1

                        # note the sides of the children
                        node.leftChild.side = 0
                        node.rightChild.side = 1

                        if len(cset) != self.setSize:
                            if node.leftChild.pathProb >= node.rightChild.pathProb:
                                heapq.heappush(L, node.rightChild)
                                node = node.leftChild
                            else:
                                heapq.heappush(L, node.leftChild)
                                node = node.rightChild
                        else:
                            heapq.heappush(L, node.leftChild)
                            heapq.heappush(L, node.rightChild)

        featureType = self.getFeatureType()
        arrLen = len(arrayStructs)
        # kh.chen
        #print("Get ArrayLenType")
        #print(self.getArrayLenType(len(arrayStructs)))

        cppCode = "{namespace}_Node{treeID} const tree{treeID}[{N}] = {" \
                .replace("{treeID}", str(treeID)) \
                .replace("{N}", str(len(arrayStructs))) \
                .replace("{namespace}", self.namespace)

        for e in arrayStructs:
                cppCode += "{"
                for val in e:
                        cppCode += str(val) + ","
                cppCode = cppCode[:-1] + "},"
        cppCode = cppCode[:-1] + "};"
        cppCode += """
                inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){
                            {arrayLenDataType} i = 0;

                            while(true) {
                                if (pX[tree{treeID}[i].feature] <= tree{treeID}[i].split){
                                    if (tree{treeID}[i].indicator == 0 || tree{treeID}[i].indicator == 2) {
                                        i = tree{treeID}[i].leftChild;
                                    } else {
                                        return tree{treeID}[i].leftChild;
                                    }
                                } else {
                                    if (tree{treeID}[i].indicator == 0 || tree{treeID}[i].indicator == 1) {
                                        i = tree{treeID}[i].rightChild;
                                    } else {
                                        return tree{treeID}[i].rightChild;
                                    }
                                }
                            }
                            return 0; // Make the compiler happy
                    }
        """.replace("{treeID}", str(treeID)) \
           .replace("{dim}", str(self.dim)) \
           .replace("{namespace}", self.namespace) \
           .replace("{arrayLenDataType}",self.getArrayLenType(len(arrayStructs))) \
           .replace("{feature_t}", featureType)

        return cppCode, arrLen


class OptimizedNativeTreeConverterForest(NativeTreeConverter):
    def __init__(self, dim, namespace, featureType, setSize = 3):
        super().__init__(dim, namespace, featureType)
        self.setSize = setSize # is this tau ?
    # call this function to get an implementation of a tree
    # optimized with alg 2
    # it gets head node, and treeID
    #
    #def getImplementation(self, head, treeID):
    #overwrite
    def getHeader(self, splitType, arrLen):
            dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

            if dimBit <= 8:
                    dimDataType = "unsigned char"
            elif dimBit <= 16:
                    dimDataType = "unsigned short"
            else:
                    dimDataType = "unsigned int"

            featureType = self.getFeatureType()
            headerCode = """struct {namespace}_Node {
                    //bool isLeaf;
                    //unsigned int prediction;
                    {dimDataType} feature;
                    {splitType} split;
                    {arrayLenDataType} leftChild;
                    {arrayLenDataType} rightChild;
                    unsigned char indicator;

            };\n""".replace("{namespace}", self.namespace) \
                       .replace("{arrayLenDataType}", self.getArrayLenType(arrLen)) \
                       .replace("{splitType}",splitType) \
                       .replace("{dimDataType}",dimDataType)

            return headerCode

    # Overwrite getCode of superclass
    def getCode(self, forest):
            # kh.chen
            # Note: this function has to be called once to traverse the tree to calculate the probabilities.
            # tree.getProbAllPaths()
            #
            cppCode, arrLen = self.getImplementation(forest)

            # We only check the data types and sizes of the first tree in the forest to decide the types and data types of all trees in the forest
            tree = forest.trees[0]

            if self.containsFloat(tree):
                splitDataType = "float"
            else:
                lower, upper = self.getSplitRange(tree)

                bitUsed = 0
                if lower > 0:
                    prefix = "unsigned"
                    maxVal = upper
                else:
                    prefix = ""
                    bitUsed = 1
                    maxVal = max(-lower, upper)

                splitBit = int(np.log2(maxVal) + 1 if maxVal != 0 else 1)

                if splitBit <= (8-bitUsed):
                    splitDataType = prefix + " char"
                elif splitBit <= (16-bitUsed):
                    splitDataType = prefix + " short"
                else:
                    splitDataType = prefix + " int"
            headerCode = self.getHeader(splitDataType, arrLen)

            return headerCode, cppCode

            #overwrite function of superclass
    def getImplementation(self, forest):
        arrayStructs = []
        nextIndexInArray = 1
        posOfRootsInArray = []
        L = []

        # put all roots in L
        for i in range(len(forest.trees)):
            # why don't we use return vals
            forest.trees[i].getProbAllPaths()
            currentHead = forest.trees[i].head
            currentHead.parent = -1
            L.append(currentHead)

        heapq.heapify(L)
        while len(L) > 0:
                #the one with the maximum probability will be the next sub-root.
                node = heapq.heappop(L)
                #print("subroot:"+str(node.id))
                cset = []
                while len(cset) != self.setSize: # 32/10
                    # cset.append(node)
                    entry = []

                    if node.prediction is not None:
                        break
                    else:
                        cset.append(node)
                        #print("split:"+str(node.id))
                        #entry.append(0)
                        #entry.append(0) # Constant prediction
                        #entry.append(node.id)
                        entry.append(node.feature)
                        entry.append(node.split)

                        if (node.leftChild.prediction is not None) and (node.rightChild.prediction is not None):
                            indicator = 3
                            # entry.append(int(node.leftChild.prediction))
                            # entry.append(int(node.rightChild.prediction))
                            entry.append(int(np.argmax(node.leftChild.prediction)))
                            entry.append(int(np.argmax(node.rightChild.prediction)))
                        elif (node.leftChild.prediction is None) and (node.rightChild.prediction is not None):
                            indicator = 2
                            entry.append(-1)
                            node.leftChild.parent = nextIndexInArray - 1

                            # entry.append(int(node.rightChild.prediction))
                            entry.append(int(np.argmax(node.rightChild.prediction)))
                        elif (node.leftChild.prediction is not None) and (node.rightChild.prediction is  None):
                            indicator = 1
                            # entry.append(int(node.leftChild.prediction))
                            entry.append(int(np.argmax(node.leftChild.prediction)))
                            entry.append(-1)
                            node.rightChild.parent = nextIndexInArray - 1

                        else:
                            indicator = 0
                            entry.append(-1)
                            node.leftChild.parent = nextIndexInArray - 1
                            entry.append(-1)
                            node.rightChild.parent = nextIndexInArray - 1
                        entry.append(indicator)

                        # node.leftChild.parent = nextIndexInArray - 1
                        # node.rightChild.parent = nextIndexInArray - 1
                        if node.parent != -1:
                            # if this node is not root, it must be assigned with self.side
                            if node.side == 0:
                                arrayStructs[node.parent][2] = nextIndexInArray - 1
                            else:
                                arrayStructs[node.parent][3] = nextIndexInArray - 1

                        # the following two fields now are modified by its children.
                        # entry.append(-1)
                        # entry.append(-1)
                        arrayStructs.append(entry)

                        # check if appended entry is root
                        # if it is root, store its index
                        if node.parent == -1:
                            posOfRootsInArray.append(nextIndexInArray-1)
                        nextIndexInArray += 1

                        # note the sides of the children
                        node.leftChild.side = 0
                        node.rightChild.side = 1

                        if len(cset) != self.setSize:
                            if node.leftChild.pathProb >= node.rightChild.pathProb:
                                heapq.heappush(L, node.rightChild)
                                node = node.leftChild
                            else:
                                heapq.heappush(L, node.leftChild)
                                node = node.rightChild
                        else:
                            heapq.heappush(L, node.leftChild)
                            heapq.heappush(L, node.rightChild)

        featureType = self.getFeatureType()
        arrLen = len(arrayStructs)
        # kh.chen
        #print("Get ArrayLenType")
        #print(self.getArrayLenType(len(arrayStructs)))
        cppCode = "unsigned int nodePos[{nrOfTrees}] = {" \
                .replace("{nrOfTrees}", str(len(posOfRootsInArray)))
        for pos in posOfRootsInArray:
            cppCode += str(pos) + ","
        cppCode = cppCode[:-1] + "};\n"


        cppCode += "{namespace}_Node const tree[{N}] = {" \
                .replace("{N}", str(len(arrayStructs))) \
                .replace("{namespace}", self.namespace)

        # Code for all nodes in forest
        for e in arrayStructs:
                cppCode += "{"
                for val in e:
                        cppCode += str(val) + ","
                cppCode = cppCode[:-1] + "},"
        cppCode = cppCode[:-1] + "};"

        return cppCode, arrLen

    # OLD code, apply alg 2 at one tree at a time
    def getImplementationOLD(self, forest):
        arrayStructs = []
        nextIndexInArray = 1
        posOfRootsInArray = []

        for i in range(len(forest.trees)):
            # store the position of roots in an array
            posOfRootsInArray.append(len(arrayStructs))

            tree = forest.trees[i]

            tree.getProbAllPaths()
            head = tree.head
            # Path-oriented Layout
            head.parent = -1 #for root init
            L = [head]
            heapq.heapify(L)
            while len(L) > 0:
                    #the one with the maximum probability will be the next sub-root.
                    node = heapq.heappop(L)
                    #print("subroot:"+str(node.id))
                    cset = []
                    while len(cset) != self.setSize: # 32/10
                        # cset.append(node)
                        entry = []

                        if node.prediction is not None:
                            break
                        else:
                            cset.append(node)
                            #print("split:"+str(node.id))
                            #entry.append(0)
                            #entry.append(0) # Constant prediction
                            #entry.append(node.id)
                            entry.append(node.feature)
                            entry.append(node.split)

                            if (node.leftChild.prediction is not None) and (node.rightChild.prediction is not None):
                                indicator = 3
                                # entry.append(int(node.leftChild.prediction))
                                # entry.append(int(node.rightChild.prediction))
                                entry.append(int(np.argmax(node.leftChild.prediction)))
                                entry.append(int(np.argmax(node.rightChild.prediction)))
                            elif (node.leftChild.prediction is None) and (node.rightChild.prediction is not None):
                                indicator = 2
                                entry.append(-1)
                                node.leftChild.parent = nextIndexInArray - 1

                                # entry.append(int(node.rightChild.prediction))
                                entry.append(int(np.argmax(node.rightChild.prediction)))
                            elif (node.leftChild.prediction is not None) and (node.rightChild.prediction is  None):
                                indicator = 1
                                # entry.append(int(node.leftChild.prediction))
                                entry.append(int(np.argmax(node.leftChild.prediction)))
                                entry.append(-1)
                                node.rightChild.parent = nextIndexInArray - 1

                            else:
                                indicator = 0
                                entry.append(-1)
                                node.leftChild.parent = nextIndexInArray - 1
                                entry.append(-1)
                                node.rightChild.parent = nextIndexInArray - 1
                            entry.append(indicator)


                            # node.leftChild.parent = nextIndexInArray - 1
                            # node.rightChild.parent = nextIndexInArray - 1
                            if node.parent != -1:
                                # if this node is not root, it must be assigned with self.side
                                if node.side == 0:
                                    arrayStructs[node.parent][2] = nextIndexInArray - 1
                                else:
                                    arrayStructs[node.parent][3] = nextIndexInArray - 1

                            # the following two fields now are modified by its children.
                            # entry.append(-1)
                            # entry.append(-1)
                            arrayStructs.append(entry)
                            nextIndexInArray += 1

                            # note the sides of the children
                            node.leftChild.side = 0
                            node.rightChild.side = 1

                            if len(cset) != self.setSize:
                                if node.leftChild.pathProb >= node.rightChild.pathProb:
                                    heapq.heappush(L, node.rightChild)
                                    node = node.leftChild
                                else:
                                    heapq.heappush(L, node.leftChild)
                                    node = node.rightChild
                            else:
                                heapq.heappush(L, node.leftChild)
                                heapq.heappush(L, node.rightChild)

        featureType = self.getFeatureType()
        arrLen = len(arrayStructs)
        # kh.chen
        #print("Get ArrayLenType")
        #print(self.getArrayLenType(len(arrayStructs)))
        cppCode = "unsigned int nodePos[{nrOfTrees}] = {" \
                .replace("{nrOfTrees}", str(len(posOfRootsInArray)))
        for pos in posOfRootsInArray:
            cppCode += str(pos) + ","
        cppCode = cppCode[:-1] + "};\n"


        cppCode += "{namespace}_Node const tree[{N}] = {" \
                .replace("{N}", str(len(arrayStructs))) \
                .replace("{namespace}", self.namespace)

        # Code for all nodes in forest
        for e in arrayStructs:
                cppCode += "{"
                for val in e:
                        cppCode += str(val) + ","
                cppCode = cppCode[:-1] + "},"
        cppCode = cppCode[:-1] + "};"

        return cppCode, arrLen
