from ForestConverter import TreeConverter
import numpy as np
import heapq

class MixConverter(TreeConverter):
        """ A MixConverter converts a DecisionTree into its mixed structure in c language
        """
        def __init__(self, dim, namespace, featureType, architecture):
                super().__init__(dim, namespace, featureType)
                #Generates a new mix-tree converter object
                self.architecture = architecture
                self.arrayLenBit = 0

                if self.architecture != "arm" and self.architecture != "intel":
                    raise NotImplementedError("Please use 'arm' or 'intel' as target architecture - other architectures are not supported")
                else:
                    if self.architecture == "arm":
                        self.setSize = 8
                    else:
                        self.setSize = 10
                self.inKernel = {}
        # size of i-cache is 32kB. One instruction is 32B. So there are 1024 instructions in i-cache
                self.givenBudget = 32*1000 # This was 32*500

        def getNativeBasis(self, head, treeID):
                return self.getNativeImplementation(head, treeID)
        def pathSort(self, tree):
            self.inKernel = {}
            curSize = 0
            s = []
            flag = False
            paths = sorted(self.getPaths(tree.head, [], []),key = lambda i: i[-1:])
            #O(n^2) to clean up the duplicates
            for i in paths:
                if i not in s:
                    s.append(i)
            for path in s:
                for node in path:
                    try:
                        if self.inKernel[node] == True:
                            continue
                    except KeyError:
                        curSize += self.sizeOfNode(tree, tree.nodes[node])
                    if curSize >= self.givenBudget:
                        self.inKernel[node] = False
                    else:
                        self.inKernel[node] = True

        def nodeSort(self, tree):
            self.inKernel = {}
            curSize = 0
            L = []
            heapq.heapify(L)
            nodes = [tree.head]
            while len(nodes) > 0:
                node = nodes.pop(0)
                if node.leftChild is not None:
                    nodes.append(node.leftChild)

                if node.rightChild is not None:
                    nodes.append(node.rightChild)
                heapq.heappush(L, node)
            # now L has BFS nodes sorted by probabilities
            while len(L) > 0:
                node = heapq.heappop(L)
                self.inKernel[node.id] = True
                curSize += self.sizeOfNode(tree,node)
                # if the current size is larger than budget already, break.
                if curSize >= self.givenBudget:
                    self.inKernel[node.id] = False
                    # in fact this can be avoided

        def sizeOfNode(self, tree, node):
            size = 0
            if self.containsFloat(tree):
                splitDataType = "float"
            else:
                splitDataType = "int"

            if node.prediction is not None:
                if splitDataType == "int" and self.architecture == "arm":
                    size += 2*4
                elif splitDataType == "float" and self.architecture == "arm":
                    size += 2*4
                elif splitDataType == "int" and self.architecture == "intel":
                    size += 10
                elif splitDataType == "float" and self.architecture == "intel":
                    size += 10
            else:
                if self.containsFloat(tree):
                    splitDataType = "float"
                else:
                    splitDataType = "int"
                # In O0, the basic size of a split node is 4 instructions for loading.
                # Since a split node must contain a pair of if-else statements,
                # one instruction for branching is not avoidable.
                if splitDataType == "int" and self.architecture == "arm":
                    # this is for arm int (ins * bytes)
                    size += 5*4
                elif splitDataType == "float" and self.architecture == "arm":
                    # this is for arm float
                    size += 8*4
                elif splitDataType == "int" and self.architecture == "intel":
                    # this is for intel integer (bytes)
                    size += 28
                elif splitDataType == "float" and self.architecture == "intel":
                    # this is for intel float (bytes)
                    size += 17
            return size


        def getIFImplementation(self, tree, treeID, head, mapping, level = 1):
        #def getIFImplementation(self, tree, treeID, head, inSize, mapping, level = 1):
            # NOTE: USE self.setSize for INTEL / ARM sepcific set-size parameter (e.g. 3 or 6)

            """ Generate the actual if-else implementation for a given node with Swapping and Kernel Grouping

            Args:
                tree : the body of this tree
                treeID (TYPE): The id of this tree (in case we are dealing with a forest)
                head (TYPE): The current node to generate an if-else structure for.
                level (int, optional): The intendation level of the generated code for easier
                                                            reading of the generated code

            Returns:
                Tuple: The string of if-else code, the string of label if-else code, generated code size and Final label index
            """
            featureType = self.getFeatureType()
            headerCode = "unsigned int {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                            .replace("{treeID}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType)
            code = ""
            tabs = "".join(['\t' for i in range(level)])
            # size of i-cache is 32kB. One instruction is 32B. So there are 1024 instructions in i-cache

            # khchen: swap-algorithm + kernel grouping
            if head.prediction is not None:
                    if self.inKernel[head.id] is False:
                        #it must be in Kernel, otherwise we are in trouble.
                        code += tabs + '\t' + "subroot = "+str(mapping[head.id])+";\n"
                        code += tabs + '\t' + "goto Label"+str(treeID)+";\n"

                    else:
                        return (tabs + "return " + str(int(head.prediction)) + ";\n" )
            else:
                    # check if it is the moment to go out the kernel, set up the root id then goto the end of the while loop.
                    if self.inKernel[head.id] is False:
                        # set up the index before goto
                        code += tabs + '\t' + "subroot = "+str(mapping[head.id])+";\n"
                        code += tabs + '\t' + "goto Label"+str(treeID)+";\n"
                    else:
                        if head.probLeft >= head.probRight:
                                code += tabs + "if(pX[" + str(head.feature) + "] <= " + str(head.split) + "){\n"
                                code += self.getIFImplementation(tree, treeID, head.leftChild,  mapping, level + 1)
                                code += tabs + "} else {\n"
                                code += self.getIFImplementation(tree, treeID, head.rightChild,  mapping, level + 1)
                        else:
                                code += tabs + "if(pX[" + str(head.feature) + "] > " + str(head.split) + "){\n"
                                code += self.getIFImplementation(tree, treeID, head.rightChild,  mapping, level + 1)
                                code += tabs + "} else {\n"
                                code += self.getIFImplementation(tree, treeID, head.leftChild,  mapping, level + 1)
                        code += tabs + "}\n"
            return (code)

        def getNativeImplementation(self, head, treeID):
            arrayStructs = []
            nextIndexInArray = 1

            mapping = {}

            # Path-oriented Layout
            head.parent = -1 #for root init
            L = [head]
            heapq.heapify(L)
            while len(L) > 0:
                    #print()
                    #for node in L:
                    #    print(node.pathProb)
                    #the one with the maximum probability will be the next sub-root.
                    node = heapq.heappop(L)
                    cset = []
                    while len(cset) != self.setSize: # 32/10
                        cset.append(node)
                        entry = []
                        mapping[node.id] = len(arrayStructs)

                        if node.prediction is not None:
                                entry.append(1)
                                entry.append(int(node.prediction))
                                entry.append(0)
                                entry.append(0)
                                entry.append(0)
                                entry.append(0)
                                arrayStructs.append(entry)

                                if node.parent != -1:
                                    # if this node is not root, it must be assigned with self.side
                                    if node.side == 0:
                                        arrayStructs[node.parent][4] = nextIndexInArray - 1
                                    else:
                                        arrayStructs[node.parent][5] = nextIndexInArray - 1


                                nextIndexInArray += 1


                                if len(L) != 0 and len(cset) != self.setSize:
                                    node = heapq.heappop(L)
                                else:
                                    break
                        else:
                                entry.append(0)
                                entry.append(0) # COnstant prediction
                                entry.append(node.feature)
                                entry.append(node.split)

                                node.leftChild.parent = nextIndexInArray - 1
                                node.rightChild.parent = nextIndexInArray - 1

                                if node.parent != -1:
                                    # if this node is not root, it must be assigned with self.side
                                    if node.side == 0:
                                        arrayStructs[node.parent][4] = nextIndexInArray - 1
                                    else:
                                        arrayStructs[node.parent][5] = nextIndexInArray - 1

                                # the following two fields now are modified by its children.
                                entry.append(-1)
                                entry.append(-1)
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

            cppCode = "{namespace}_Node{id} const tree{id}[{N}] = {" \
                    .replace("{id}", str(treeID)) \
                    .replace("{N}", str(len(arrayStructs))) \
                    .replace("{namespace}", self.namespace)

            for e in arrayStructs:
                    cppCode += "{"
                    for val in e:
                            cppCode += str(val) + ","
                    cppCode = cppCode[:-1] + "},"
            cppCode = cppCode[:-1] + "};"

            return cppCode, arrLen, mapping


        def getMaxThreshold(self, tree):
                return max([tree.nodes[x].split if tree.nodes[x].prediction is None else 0 for x in tree.nodes])

        def getArrayLenType(self, arrLen):
                arrayLenBit = int(np.log2(arrLen)) + 1
                if arrayLenBit <= 8:
                        arrayLenDataType = "unsigned char"
                elif arrayLenBit <= 16:
                        arrayLenDataType = "unsigned short"
                else:
                        arrayLenDataType = "unsigned int"
                return arrayLenDataType


        def getNativeHeader(self, splitType, treeID, arrLen):
                dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

                if dimBit <= 8:
                        dimDataType = "unsigned char"
                elif dimBit <= 16:
                        dimDataType = "unsigned short"
                else:
                        dimDataType = "unsigned int"

                featureType = self.getFeatureType()
                headerCode = """struct {namespace}_Node{id} {
                        bool isLeaf;
                        unsigned int prediction;
                        {dimDataType} feature;
                        {splitType} split;
                        {arrayLenDataType} leftChild;
                        {arrayLenDataType} rightChild;

                };\n""".replace("{namespace}", self.namespace) \
                           .replace("{id}", str(treeID)) \
                           .replace("{arrayLenDataType}", self.getArrayLenType(arrLen)) \
                           .replace("{splitType}",splitType) \
                           .replace("{dimDataType}",dimDataType)
                """
                headerCode += "unsigned int {namespace}_predict{id}({feature_t} const pX[{dim}]);\n" \
                                                .replace("{id}", str(treeID)) \
                                                .replace("{dim}", str(self.dim)) \
                                                .replace("{namespace}", self.namespace) \
                                                .replace("{feature_t}", featureType)
                """
                return headerCode


        def getCode(self, tree, treeID):
            """ Generate the actual mixture implementation for a given tree

            Args:
                tree (TYPE): The tree
                treeID (TYPE): The id of this tree (in case we are dealing with a forest)

            Returns:
                Tuple: A tuple (headerCode, cppCode), where headerCode contains the code (=string) for
                a *.h file and cppCode contains the code (=string) for a *.cpp file
            """
            tree.getProbAllPaths()
            featureType = self.getFeatureType()
            cppCode = "unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){\n" \
                                    .replace("{treeID}", str(treeID)) \
                                    .replace("{dim}", str(self.dim)) \
                                    .replace("{namespace}", self.namespace) \
                                    .replace("{feature_t}", featureType)
            cppCode += "unsigned int subroot;\n"
            nativeImplementation = self.getNativeImplementation(tree.head, treeID)
            cppCode += nativeImplementation[0]+ "\n"
            arrLen = nativeImplementation[1]
            mapping = nativeImplementation[2]

            self.nodeSort(tree)
            ifImplementation = self.getIFImplementation(tree, treeID, tree.head, mapping, 0)
            # kernel code
            cppCode += ifImplementation

            # Data Array
            cppCode += """
                    Label{id}:
                    {
                            {arrayLenDataType} i = subroot;

                            while(!tree{id}[i].isLeaf) {
                                    if (pX[tree{id}[i].feature] <= tree{id}[i].split){
                                            i = tree{id}[i].leftChild;
                                    } else {
                                            i = tree{id}[i].rightChild;
                                    }
                            }

                            return tree{id}[i].prediction;
                    }
            """.replace("{id}", str(treeID)) \
               .replace("{arrayLenDataType}",self.getArrayLenType(arrLen))

            cppCode += "}\n"

            # the rest is for generating the header
            headerCode = "unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                            .replace("{treeID}", str(treeID)) \
                                            .replace("{dim}", str(self.dim)) \
                                            .replace("{namespace}", self.namespace) \
                                            .replace("{feature_t}", featureType)

            if self.containsFloat(tree):
                splitDataType = "float"
            else:
                lower, upper = self.getSplitRange(tree)

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

            headerCode += self.getNativeHeader(splitDataType, treeID, arrLen)
            return headerCode, cppCode
