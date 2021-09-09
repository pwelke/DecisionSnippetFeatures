# DecisionSnippetFeatures

Code and data accompanying the paper

Pascal Welke, Fouad Alkhoury, Christian Bauckhage, and Stefan Wrobel: 
[Decision Snippet Features](https://mlai.cs.uni-bonn.de/publications/welke2021-dsf.pdf).
25th International Conference on Pattern Recognition (ICPR) 2021, Milano, Italy.
[DOI:10.1109/ICPR48806.2021.9412025](http://dx.doi.org/10.1109/ICPR48806.2021.9412025)

If you use this code, please [cite our paper](https://dblp.uni-trier.de/rec/conf/icpr/WelkeABW20.html?view=bibtex).

## Requirements

To install the necessary requirements, we provide a ```environment.yml``` file [that can be used with anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
You may of course use any other means of installing the requirements listed in this file.

It is possible that this code will continue to work with more recent versions of the requirements but we don't guarantee anything.

## How to run the code

The file ```/dsf/program.py``` is the entry point to the decision snippet pipeline. 
In order to keep the code rather concise, we don't include timing measurements, as well as the code to measure average number of inference steps.

In order to change something in the code, you may have a look at the parameters section in ```/dsf/program.py```:

    ```python
    dataPath = "./data/"
    forestsPath = "./tmp/forests/"
    snippetsPath = "./tmp/snippets/"
    resultsPath = "./tmp/results/"


    # current valid options are ['sensorless', 'satlog', 'mnist', 'magic', 'spambase', 'letter', 'bank', 'adult', 'drinking']
    dataSet = 'magic'
    # dataSet = 'adult'
    # dataSet = 'drinking'

    # possible forest_types ['RF', 'DT', 'ET']
    forest_types = ['RF']
    forest_depths = [5, 10, 15, 20]
    forest_size = 25

    maxPatternSize = 6
    minThreshold = 2
    maxThreshold = 25

    scoring_function = 'accuracy'

    # learners that are to be used on top of Decision Snippet Features
    learners = {'DSF_NB': MultinomialNB,
                'DSF_SVM': LinearSVC, 
                'DSF_LR': LogisticRegression}

    # specify parameters that are given at initialization
    learners_parameters = {'DSF_NB': {},
                        'DSF_SVM': {'max_iter': 10000},
                        'DSF_LR': {'max_iter': 1000}}


    # for quick debugging, let the whole thing run once. Afterwards, you may deactivate individual steps
    # each step stores its output for the subsequent step(s) to process
    run_fit_models = True
    run_mining = True
    run_training = True
    run_eval = True

    verbose = True
    ```

Datasets are provided in ```/data/``` folder and can be selected in the parameter section of ```/dsf/program.py``` by changing ```dataSet``` accordingly.
The code ```/dsf/program.py``` is intended to be run for a single dataset on each call. 

Output will be written to the folders specified by ```*Path``` variables. 
For each type of output, a subfolder with the name of the current dataset will be created.



## Comments

Please note that the code will not create the exact numbers reported in the paper when you run it. 
This is due to randomization in the Random Forests, as well as random cross validation splits when selecting the best Decision Snippet Features for each Learner. 

Finally, I would like to thank [Lukas Pfahler](https://github.com/Whadup) to point out an error in an earlier version of our code.
