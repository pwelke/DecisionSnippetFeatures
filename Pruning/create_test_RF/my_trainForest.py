#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np

from my_fitModels import fitModels

np.random.seed(5)

def main(argv):
    data = np.genfromtxt("test_dataset.csv", delimiter=';', skip_header=1)
    X = data[:,:-1].astype(np.int)
    Y = data[:,-1].astype(np.int)
    #X = np.random.randint(1,11,(10,3))  #generate a new random dataset
    #Y = np.random.randint(0,2,(10,))
    print(X)
    print(Y)
    fitModels(False,X,Y)


if __name__ == "__main__":
   main(sys.argv[1:])
