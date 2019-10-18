# Architectural specific Random Forests
### Fast Random Forest traversal through efficient memory layout 

---

This repository contains the code for our ICDM 2018 submission "Realization of Random Forest for Real-Time Evaluation through Tree Framing". To reproduce the exact state of the experiments you can checkout the following tag:

First, clone the repo ``git clone git@bitbucket.org:sbuschjaeger/arch-forest.git``
Then, change the folder ``cd arch-forest``
Third, checkout the following tag ``git checkout tags/v1.1``

Alternatively you can use the latest version of the code. However, I cannot guarantee that the latest version will run all the time. 

The experiments come with a number of dependencies depending on the parts you wish to use:

* For running the code generator you will need ``python3`` and ``numpy`` (http://www.numpy.org/)
* For training new models, you will need ``scikit-learn`` (http://scikit-learn.org/stable/)
* For running the experiments, on intel you need ``g++`` (We used g++ 5.4.1 and g++ 6.3.0 during the experiments. However, any g++ version supporting c++11 should be fine)
* For running the experiments, on arm you need ``arm-linux-gnueabihf-g++``  (We used g++ 4.8.4 and g++ 6.3.0 during the experiments. However, any g++ version supporting c++11 should be fine)  

The experiments were performed on Linux systems. There are different bash-scripts which are called to perform certain parts of the experiments. These script usually come with additionally requirements such as ``wget`` or ``stat`` which should be available. However, any remotely-standard linux installation should offer these.

A simple how-to is listed below. Otherwise, the basic structure is as the following:

* ``code/`` contains the actual forest and tree code synthesizer discussed in the paper  
* ``data/`` contains scripts and files for running the experiments. Each folder represents one data set used in the experiments. There are a couple of scripts for convienience. Let ``dataset`` be a dataset of choice, then
    * ``dataset/init.sh`` can be used to download and prepare this dataset. Please note, that not all data-sets can be directly downloaded via script (``imdb``,``fact``,``trec``). Please download those manually. The URL can be found in the init-script. Also note, that ``wearable-body-postures`` needs some manual editing of the training data, because there is a wrong line in the original file.
    * ``dataset/trainForest.py`` This trains a new RF with 25 trees on the corresponding dataset using ``sklearn`` and stores the trained model as JSON file in ``dataset/text/forest_25.json``. Additionally, the model is exported as python pickle file in ``dataset/text/forsest_25.pkl``
    * ``generateCode.py`` This script does the actual code generation. It receives 2 parameters. The first parameter is dataset for which code should be generated, the second one is the target architecture (``arm`` or ``intel``). This will generate the necessary  cpp files for testing and generate a Makefile for compilation in the ``dataset/cpp/architechture/modelname `` folder. 
    * ``compile.sh`` This script receives two parameters. It will compile the cpp files for the given dataset (first parameter) and target architecture (second parameter). Please make sure, that the necessary compiler is installed on your system. For intel we use ``g++``. For arm ``arm-linux-gnueabihf-g++`` is used. 
    * ``run.sh`` This script receives two parameters. It will run the compiled cpp files for the given dataset (first parameter) and target architecture (second parameter). Results will be printed to std out. 
      ``runSKLearn.sh`` This script receives one parameter. It receives a folder and  will load the stored SKLearn model file (from the ``text`` folder) and run it on the corresponding dataset. Results will be printed to std out.
    * ``init_all.sh`` This will call the ``init.sh`` script on all folders
    * ``generate_all.sh`` This will call the ``generateCode.py`` script on all folders. It receives the target architecture as parameter
    * ``compile_all.sh`` This will call the ``compile.sh`` script on all folders.It receives the target architecture as parameter
    * ``run_all.sh`` This will call the ``run.sh`` script on all folders.It receives the target architecture as parameter


#How to

```bash
# Download the data
cd wine-quality
./init.sh

# Train the model
./trainForest.py

# Generate the code for intel
cd ..
./generateCode wine-quality intel

# Compile it
./compile.sh wine-quality intel

# Run it
./run.sh wine-quality intel
```