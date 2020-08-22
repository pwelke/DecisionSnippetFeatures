# Frequent Subtree Counting in Random Forests

Similar to the notebook 

    Initial Rooted Frequent Subtree Mining (without embedding computation).ipnb
    
I will start the same mining / evaluation process for the data, but will include the split value in the labels, 
i.e., graph vertices now are labeled 
    
    'ID<NUM'
    

## Datasets
There are several datasets.
At the moment, however, I'll experiment only with 'adult' and 'wine-quality'.

## Find Frequent Rooted Trees

Let's see how many rooted frequent trees we can find in the random forests.


```bash
%%bash
for dataset in adult wine-quality; do
    for variant in NoLeafEdgesWithSplitValues; do
        mkdir forests/${dataset}/${variant}/
    done
done
```

    mkdir: cannot create directory ‘forests/adult/NoLeafEdgesWithSplitValues/’: File exists
    mkdir: cannot create directory ‘forests/wine-quality/NoLeafEdgesWithSplitValues/’: File exists



    ---------------------------------------------------------------------------

    CalledProcessError                        Traceback (most recent call last)

    <ipython-input-1-c7278461b8ad> in <module>
    ----> 1 get_ipython().run_cell_magic('bash', '', 'for dataset in adult wine-quality; do\n    for variant in NoLeafEdgesWithSplitValues; do\n        mkdir forests/${dataset}/${variant}/\n    done\ndone\n')
    

    /home/iai/user/blatt/myconda/bigenv/lib/python3.8/site-packages/IPython/core/interactiveshell.py in run_cell_magic(self, magic_name, line, cell)
       2369             with self.builtin_trap:
       2370                 args = (magic_arg_s, cell)
    -> 2371                 result = fn(*args, **kwargs)
       2372             return result
       2373 


    /home/iai/user/blatt/myconda/bigenv/lib/python3.8/site-packages/IPython/core/magics/script.py in named_script_magic(line, cell)
        140             else:
        141                 line = script
    --> 142             return self.shebang(line, cell)
        143 
        144         # write a basic docstring:


    <decorator-gen-110> in shebang(self, line, cell)


    /home/iai/user/blatt/myconda/bigenv/lib/python3.8/site-packages/IPython/core/magic.py in <lambda>(f, *a, **k)
        185     # but it's overkill for just that one bit of state.
        186     def magic_deco(arg):
    --> 187         call = lambda f, *a, **k: f(*a, **k)
        188 
        189         if callable(arg):


    /home/iai/user/blatt/myconda/bigenv/lib/python3.8/site-packages/IPython/core/magics/script.py in shebang(self, line, cell)
        243             sys.stderr.flush()
        244         if args.raise_error and p.returncode!=0:
    --> 245             raise CalledProcessError(p.returncode, cell, output=out, stderr=err)
        246 
        247     def _run_script(self, p, cell, to_close):


    CalledProcessError: Command 'b'for dataset in adult wine-quality; do\n    for variant in NoLeafEdgesWithSplitValues; do\n        mkdir forests/${dataset}/${variant}/\n    done\ndone\n'' returned non-zero exit status 1.



```bash
%%bash
for dataset in adult wine-quality; do
    for f in forests/${dataset}/text/*.json; do
        echo ${f} '->' `basename ${f} .json`.graph
        python json2graphNoLeafEdgesWithSplitValues.py ${f} > forests/${dataset}/NoLeafEdgesWithSplitValues/`basename ${f} .json`.graph
    done
done
```

    forests/adult/text/DT_10.json -> DT_10.graph
    forests/adult/text/DT_10_pruned_with_sigma_0_0.json -> DT_10_pruned_with_sigma_0_0.graph
    forests/adult/text/DT_10_pruned_with_sigma_0_1.json -> DT_10_pruned_with_sigma_0_1.graph
    forests/adult/text/DT_10_pruned_with_sigma_0_2.json -> DT_10_pruned_with_sigma_0_2.graph
    forests/adult/text/DT_10_pruned_with_sigma_0_3.json -> DT_10_pruned_with_sigma_0_3.graph
    forests/adult/text/DT_15.json -> DT_15.graph
    forests/adult/text/DT_15_pruned_with_sigma_0_0.json -> DT_15_pruned_with_sigma_0_0.graph
    forests/adult/text/DT_15_pruned_with_sigma_0_1.json -> DT_15_pruned_with_sigma_0_1.graph
    forests/adult/text/DT_15_pruned_with_sigma_0_2.json -> DT_15_pruned_with_sigma_0_2.graph
    forests/adult/text/DT_15_pruned_with_sigma_0_3.json -> DT_15_pruned_with_sigma_0_3.graph
    forests/adult/text/DT_1.json -> DT_1.graph
    forests/adult/text/DT_1_pruned_with_sigma_0_0.json -> DT_1_pruned_with_sigma_0_0.graph
    forests/adult/text/DT_1_pruned_with_sigma_0_1.json -> DT_1_pruned_with_sigma_0_1.graph
    forests/adult/text/DT_1_pruned_with_sigma_0_2.json -> DT_1_pruned_with_sigma_0_2.graph
    forests/adult/text/DT_1_pruned_with_sigma_0_3.json -> DT_1_pruned_with_sigma_0_3.graph
    forests/adult/text/DT_20.json -> DT_20.graph
    forests/adult/text/DT_20_pruned_with_sigma_0_0.json -> DT_20_pruned_with_sigma_0_0.graph
    forests/adult/text/DT_20_pruned_with_sigma_0_1.json -> DT_20_pruned_with_sigma_0_1.graph
    forests/adult/text/DT_20_pruned_with_sigma_0_2.json -> DT_20_pruned_with_sigma_0_2.graph
    forests/adult/text/DT_20_pruned_with_sigma_0_3.json -> DT_20_pruned_with_sigma_0_3.graph
    forests/adult/text/DT_5.json -> DT_5.graph
    forests/adult/text/DT_5_pruned_with_sigma_0_0.json -> DT_5_pruned_with_sigma_0_0.graph
    forests/adult/text/DT_5_pruned_with_sigma_0_1.json -> DT_5_pruned_with_sigma_0_1.graph
    forests/adult/text/DT_5_pruned_with_sigma_0_2.json -> DT_5_pruned_with_sigma_0_2.graph
    forests/adult/text/DT_5_pruned_with_sigma_0_3.json -> DT_5_pruned_with_sigma_0_3.graph
    forests/adult/text/ET_10.json -> ET_10.graph
    forests/adult/text/ET_10_pruned_with_sigma_0_0.json -> ET_10_pruned_with_sigma_0_0.graph
    forests/adult/text/ET_10_pruned_with_sigma_0_1.json -> ET_10_pruned_with_sigma_0_1.graph
    forests/adult/text/ET_10_pruned_with_sigma_0_2.json -> ET_10_pruned_with_sigma_0_2.graph
    forests/adult/text/ET_10_pruned_with_sigma_0_3.json -> ET_10_pruned_with_sigma_0_3.graph
    forests/adult/text/ET_15.json -> ET_15.graph
    forests/adult/text/ET_15_pruned_with_sigma_0_0.json -> ET_15_pruned_with_sigma_0_0.graph
    forests/adult/text/ET_15_pruned_with_sigma_0_1.json -> ET_15_pruned_with_sigma_0_1.graph
    forests/adult/text/ET_15_pruned_with_sigma_0_2.json -> ET_15_pruned_with_sigma_0_2.graph
    forests/adult/text/ET_15_pruned_with_sigma_0_3.json -> ET_15_pruned_with_sigma_0_3.graph
    forests/adult/text/ET_1.json -> ET_1.graph
    forests/adult/text/ET_1_pruned_with_sigma_0_0.json -> ET_1_pruned_with_sigma_0_0.graph
    forests/adult/text/ET_1_pruned_with_sigma_0_1.json -> ET_1_pruned_with_sigma_0_1.graph
    forests/adult/text/ET_1_pruned_with_sigma_0_2.json -> ET_1_pruned_with_sigma_0_2.graph
    forests/adult/text/ET_1_pruned_with_sigma_0_3.json -> ET_1_pruned_with_sigma_0_3.graph
    forests/adult/text/ET_20.json -> ET_20.graph
    forests/adult/text/ET_20_pruned_with_sigma_0_0.json -> ET_20_pruned_with_sigma_0_0.graph
    forests/adult/text/ET_20_pruned_with_sigma_0_1.json -> ET_20_pruned_with_sigma_0_1.graph
    forests/adult/text/ET_20_pruned_with_sigma_0_2.json -> ET_20_pruned_with_sigma_0_2.graph
    forests/adult/text/ET_20_pruned_with_sigma_0_3.json -> ET_20_pruned_with_sigma_0_3.graph
    forests/adult/text/ET_5.json -> ET_5.graph
    forests/adult/text/ET_5_pruned_with_sigma_0_0.json -> ET_5_pruned_with_sigma_0_0.graph
    forests/adult/text/ET_5_pruned_with_sigma_0_1.json -> ET_5_pruned_with_sigma_0_1.graph
    forests/adult/text/ET_5_pruned_with_sigma_0_2.json -> ET_5_pruned_with_sigma_0_2.graph
    forests/adult/text/ET_5_pruned_with_sigma_0_3.json -> ET_5_pruned_with_sigma_0_3.graph
    forests/adult/text/RF_10.json -> RF_10.graph
    forests/adult/text/RF_10_pruned_with_sigma_0_0.json -> RF_10_pruned_with_sigma_0_0.graph
    forests/adult/text/RF_10_pruned_with_sigma_0_1.json -> RF_10_pruned_with_sigma_0_1.graph
    forests/adult/text/RF_10_pruned_with_sigma_0_2.json -> RF_10_pruned_with_sigma_0_2.graph
    forests/adult/text/RF_10_pruned_with_sigma_0_3.json -> RF_10_pruned_with_sigma_0_3.graph
    forests/adult/text/RF_15.json -> RF_15.graph
    forests/adult/text/RF_15_pruned_with_sigma_0_0.json -> RF_15_pruned_with_sigma_0_0.graph
    forests/adult/text/RF_15_pruned_with_sigma_0_1.json -> RF_15_pruned_with_sigma_0_1.graph
    forests/adult/text/RF_15_pruned_with_sigma_0_2.json -> RF_15_pruned_with_sigma_0_2.graph
    forests/adult/text/RF_15_pruned_with_sigma_0_3.json -> RF_15_pruned_with_sigma_0_3.graph
    forests/adult/text/RF_1.json -> RF_1.graph
    forests/adult/text/RF_1_pruned_with_sigma_0_0.json -> RF_1_pruned_with_sigma_0_0.graph
    forests/adult/text/RF_1_pruned_with_sigma_0_1.json -> RF_1_pruned_with_sigma_0_1.graph
    forests/adult/text/RF_1_pruned_with_sigma_0_2.json -> RF_1_pruned_with_sigma_0_2.graph
    forests/adult/text/RF_1_pruned_with_sigma_0_3.json -> RF_1_pruned_with_sigma_0_3.graph
    forests/adult/text/RF_20.json -> RF_20.graph
    forests/adult/text/RF_20_pruned_with_sigma_0_0.json -> RF_20_pruned_with_sigma_0_0.graph
    forests/adult/text/RF_20_pruned_with_sigma_0_1.json -> RF_20_pruned_with_sigma_0_1.graph
    forests/adult/text/RF_20_pruned_with_sigma_0_2.json -> RF_20_pruned_with_sigma_0_2.graph
    forests/adult/text/RF_20_pruned_with_sigma_0_3.json -> RF_20_pruned_with_sigma_0_3.graph
    forests/adult/text/RF_5.json -> RF_5.graph
    forests/adult/text/RF_5_pruned_with_sigma_0_0.json -> RF_5_pruned_with_sigma_0_0.graph
    forests/adult/text/RF_5_pruned_with_sigma_0_1.json -> RF_5_pruned_with_sigma_0_1.graph
    forests/adult/text/RF_5_pruned_with_sigma_0_2.json -> RF_5_pruned_with_sigma_0_2.graph
    forests/adult/text/RF_5_pruned_with_sigma_0_3.json -> RF_5_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/DT_10.json -> DT_10.graph
    forests/wine-quality/text/DT_10_pruned_with_sigma_0_0.json -> DT_10_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/DT_10_pruned_with_sigma_0_1.json -> DT_10_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/DT_10_pruned_with_sigma_0_2.json -> DT_10_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/DT_10_pruned_with_sigma_0_3.json -> DT_10_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/DT_15.json -> DT_15.graph
    forests/wine-quality/text/DT_15_pruned_with_sigma_0_0.json -> DT_15_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/DT_15_pruned_with_sigma_0_1.json -> DT_15_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/DT_15_pruned_with_sigma_0_2.json -> DT_15_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/DT_15_pruned_with_sigma_0_3.json -> DT_15_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/DT_1.json -> DT_1.graph
    forests/wine-quality/text/DT_1_pruned_with_sigma_0_0.json -> DT_1_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/DT_1_pruned_with_sigma_0_1.json -> DT_1_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/DT_1_pruned_with_sigma_0_2.json -> DT_1_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/DT_1_pruned_with_sigma_0_3.json -> DT_1_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/DT_20.json -> DT_20.graph
    forests/wine-quality/text/DT_20_pruned_with_sigma_0_0.json -> DT_20_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/DT_20_pruned_with_sigma_0_1.json -> DT_20_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/DT_20_pruned_with_sigma_0_2.json -> DT_20_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/DT_20_pruned_with_sigma_0_3.json -> DT_20_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/DT_5.json -> DT_5.graph
    forests/wine-quality/text/DT_5_pruned_with_sigma_0_0.json -> DT_5_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/DT_5_pruned_with_sigma_0_1.json -> DT_5_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/DT_5_pruned_with_sigma_0_2.json -> DT_5_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/DT_5_pruned_with_sigma_0_3.json -> DT_5_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/ET_10.json -> ET_10.graph
    forests/wine-quality/text/ET_10_pruned_with_sigma_0_0.json -> ET_10_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/ET_10_pruned_with_sigma_0_1.json -> ET_10_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/ET_10_pruned_with_sigma_0_2.json -> ET_10_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/ET_10_pruned_with_sigma_0_3.json -> ET_10_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/ET_15.json -> ET_15.graph
    forests/wine-quality/text/ET_15_pruned_with_sigma_0_0.json -> ET_15_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/ET_15_pruned_with_sigma_0_1.json -> ET_15_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/ET_15_pruned_with_sigma_0_2.json -> ET_15_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/ET_15_pruned_with_sigma_0_3.json -> ET_15_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/ET_1.json -> ET_1.graph
    forests/wine-quality/text/ET_1_pruned_with_sigma_0_0.json -> ET_1_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/ET_1_pruned_with_sigma_0_1.json -> ET_1_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/ET_1_pruned_with_sigma_0_2.json -> ET_1_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/ET_1_pruned_with_sigma_0_3.json -> ET_1_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/ET_20.json -> ET_20.graph
    forests/wine-quality/text/ET_20_pruned_with_sigma_0_0.json -> ET_20_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/ET_20_pruned_with_sigma_0_1.json -> ET_20_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/ET_20_pruned_with_sigma_0_2.json -> ET_20_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/ET_20_pruned_with_sigma_0_3.json -> ET_20_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/ET_5.json -> ET_5.graph
    forests/wine-quality/text/ET_5_pruned_with_sigma_0_0.json -> ET_5_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/ET_5_pruned_with_sigma_0_1.json -> ET_5_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/ET_5_pruned_with_sigma_0_2.json -> ET_5_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/ET_5_pruned_with_sigma_0_3.json -> ET_5_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/RF_10.json -> RF_10.graph
    forests/wine-quality/text/RF_10_pruned_with_sigma_0_0.json -> RF_10_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/RF_10_pruned_with_sigma_0_1.json -> RF_10_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/RF_10_pruned_with_sigma_0_2.json -> RF_10_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/RF_10_pruned_with_sigma_0_3.json -> RF_10_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/RF_15.json -> RF_15.graph
    forests/wine-quality/text/RF_15_pruned_with_sigma_0_0.json -> RF_15_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/RF_15_pruned_with_sigma_0_1.json -> RF_15_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/RF_15_pruned_with_sigma_0_2.json -> RF_15_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/RF_15_pruned_with_sigma_0_3.json -> RF_15_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/RF_1.json -> RF_1.graph
    forests/wine-quality/text/RF_1_pruned_with_sigma_0_0.json -> RF_1_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/RF_1_pruned_with_sigma_0_1.json -> RF_1_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/RF_1_pruned_with_sigma_0_2.json -> RF_1_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/RF_1_pruned_with_sigma_0_3.json -> RF_1_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/RF_20.json -> RF_20.graph
    forests/wine-quality/text/RF_20_pruned_with_sigma_0_0.json -> RF_20_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/RF_20_pruned_with_sigma_0_1.json -> RF_20_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/RF_20_pruned_with_sigma_0_2.json -> RF_20_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/RF_20_pruned_with_sigma_0_3.json -> RF_20_pruned_with_sigma_0_3.graph
    forests/wine-quality/text/RF_5.json -> RF_5.graph
    forests/wine-quality/text/RF_5_pruned_with_sigma_0_0.json -> RF_5_pruned_with_sigma_0_0.graph
    forests/wine-quality/text/RF_5_pruned_with_sigma_0_1.json -> RF_5_pruned_with_sigma_0_1.graph
    forests/wine-quality/text/RF_5_pruned_with_sigma_0_2.json -> RF_5_pruned_with_sigma_0_2.graph
    forests/wine-quality/text/RF_5_pruned_with_sigma_0_3.json -> RF_5_pruned_with_sigma_0_3.graph



```bash
%%bash
mkdir forests/rootedFrequentTrees
# create output directories
for dataset in adult wine-quality; do
    mkdir forests/rootedFrequentTrees/${dataset}/
    for variant in NoLeafEdgesWithSplitValues; do
        mkdir forests/rootedFrequentTrees/${dataset}/${variant}/
    done
done
```

    mkdir: cannot create directory ‘forests/rootedFrequentTrees’: File exists
    mkdir: cannot create directory ‘forests/rootedFrequentTrees/adult/’: File exists
    mkdir: cannot create directory ‘forests/rootedFrequentTrees/adult/NoLeafEdgesWithSplitValues/’: File exists
    mkdir: cannot create directory ‘forests/rootedFrequentTrees/wine-quality/’: File exists
    mkdir: cannot create directory ‘forests/rootedFrequentTrees/wine-quality/NoLeafEdgesWithSplitValues/’: File exists



    ---------------------------------------------------------------------------

    CalledProcessError                        Traceback (most recent call last)

    <ipython-input-3-1eace3437e85> in <module>
    ----> 1 get_ipython().run_cell_magic('bash', '', 'mkdir forests/rootedFrequentTrees\n# create output directories\nfor dataset in adult wine-quality; do\n    mkdir forests/rootedFrequentTrees/${dataset}/\n    for variant in NoLeafEdgesWithSplitValues; do\n        mkdir forests/rootedFrequentTrees/${dataset}/${variant}/\n    done\ndone\n')
    

    /home/iai/user/blatt/myconda/bigenv/lib/python3.8/site-packages/IPython/core/interactiveshell.py in run_cell_magic(self, magic_name, line, cell)
       2369             with self.builtin_trap:
       2370                 args = (magic_arg_s, cell)
    -> 2371                 result = fn(*args, **kwargs)
       2372             return result
       2373 


    /home/iai/user/blatt/myconda/bigenv/lib/python3.8/site-packages/IPython/core/magics/script.py in named_script_magic(line, cell)
        140             else:
        141                 line = script
    --> 142             return self.shebang(line, cell)
        143 
        144         # write a basic docstring:


    <decorator-gen-110> in shebang(self, line, cell)


    /home/iai/user/blatt/myconda/bigenv/lib/python3.8/site-packages/IPython/core/magic.py in <lambda>(f, *a, **k)
        185     # but it's overkill for just that one bit of state.
        186     def magic_deco(arg):
    --> 187         call = lambda f, *a, **k: f(*a, **k)
        188 
        189         if callable(arg):


    /home/iai/user/blatt/myconda/bigenv/lib/python3.8/site-packages/IPython/core/magics/script.py in shebang(self, line, cell)
        243             sys.stderr.flush()
        244         if args.raise_error and p.returncode!=0:
    --> 245             raise CalledProcessError(p.returncode, cell, output=out, stderr=err)
        246 
        247     def _run_script(self, p, cell, to_close):


    CalledProcessError: Command 'b'mkdir forests/rootedFrequentTrees\n# create output directories\nfor dataset in adult wine-quality; do\n    mkdir forests/rootedFrequentTrees/${dataset}/\n    for variant in NoLeafEdgesWithSplitValues; do\n        mkdir forests/rootedFrequentTrees/${dataset}/${variant}/\n    done\ndone\n'' returned non-zero exit status 1.



```bash
%%bash
./lwgr -h
```

    This is a frequent rooted subtree mining tool.
    Implemented by Pascal Welke starting in 2018.
    
    This program computes and outputs frequent *rooted* subtrees and feature
    representations of the mined graphs. The database is expected to contain
    tree transactions that are interpreted as being rooted at the first
    vertex.
    
    usage: ./lwg [options] [FILE]
    
    If no FILE argument is given or FILE is - the program reads from stdin.
    It always prints to stdout (unless specified by parameters) and 
    stderr (statistics).
    
    
    Options:
    -h:           print this possibly helpful information.
    
    -t THRESHOLD: Minimum absolute support threshold in the graph database
    
    -p SIZE:      Maximum size (number of vertices) of patterns returned
    
    -o FILE:      output the frequent subtrees in this file
    
    -f FILE:      output the feature information in this file
    
    -i VALUE:     Some embedding operators require a parameter that might be
                  a float between 0.0 and 1.0 or an integer >=1, depending 
                  on the operator.
                  
    -r VALUE:     Initialize the random number generator with seed VALUE. If
                  not specified, random generator is seeded according to 
                  current time.
    
    
    -m METHOD:    Choose mining method among
                  
                  bfs: (default) mine in a levelwise fashion (like apriori). 
                     This results in better pruning behavior, but large memory 
                     footprint
          
    
    -e OPERATOR:  Select the algorithm to decide whether a tree pattern 
                  matches a transaction graph.
                  Choose embedding operator among the following:
    
             == EXACT TREE EMBEDDING OPERATORS ==
               These operators result in the full set of frequent 
               rooted subtrees being output by this mining algorithm.
             
               rootedTrees: (default) classical subtree isomorphism algorithm.
                 A pattern matches a graph, if it is rooted subgraph isomorphic 
                 to it. That is, the transactions are interpreted as being rooted
                 at their first vertex (in the graph database file format) and the 
                 image of the root of the pattern must be the vertex in the image of 
                 the pattern with the smallest distance to the root of the transaction.
                  Works only for forest transaction databases.
    
                 
             == STRANGE EMBEDDING OPERATORS ==
               These operators do various stuff. The first two are 
               stronger than subgraph isomorphism, requiring possibly
               more than one embedding into a transaction graph to
               match the graph. The third just enumerates all trees
               up to isomorphism that can be created from the vertex 
               and edge labels in the database.
                              
               rootedTreeEnumeration: Enumerate all trees up to isomorphism which 
                 can be generated from frequent vertices and edges in the
                 input databases.
             
    
     



```bash
%%bash
rm todolist.txt
for dataset in adult wine-quality; do
    for variant in NoLeafEdgesWithSplitValues; do
        for f in forests/${dataset}/${variant}/*.graph; do
            for threshold in 2; do
                #echo "processing threshold ${threshold} for ${f}"
                echo "./lwgr -e rootedTrees -m bfs -t ${threshold} -p 10 \
                  -o forests/rootedFrequentTrees/${dataset}/${variant}/`basename ${f} .graph`_t${threshold}.patterns \
                  < ${f} \
                  > forests/rootedFrequentTrees/${dataset}/${variant}/`basename ${f} .graph`_t${threshold}.features \
                  2> forests/rootedFrequentTrees/${dataset}/${variant}/`basename ${f} .graph`_t${threshold}.logs" >> todolist.txt
                  
            done
        done
    done
done
```


```bash
%%bash
cat todolist.txt | parallel
```


```bash
%%bash
for dataset in adult wine-quality; do
    for variant in NoLeafEdgesWithSplitValues; do
        for f in forests/${dataset}/${variant}/*_20.graph; do
            threshold=2
            echo "processing threshold ${threshold} for ${f}"
            ./lwgr -e rootedTrees -m bfs -t ${threshold} -p 10 \
              -o forests/rootedFrequentTrees/${dataset}/${variant}/`basename ${f} .graph`_t${threshold}.patterns \
              < ${f} \
              > forests/rootedFrequentTrees/${dataset}/${variant}/`basename ${f} .graph`_t${threshold}.features \
              2> forests/rootedFrequentTrees/${dataset}/${variant}/`basename ${f} .graph`_t${threshold}.logs       
        done
    done
done
```

    processing threshold 2 for forests/adult/NoLeafEdgesWithSplitValues/DT_20.graph
    processing threshold 2 for forests/adult/NoLeafEdgesWithSplitValues/ET_20.graph
    processing threshold 2 for forests/adult/NoLeafEdgesWithSplitValues/RF_20.graph
    processing threshold 2 for forests/wine-quality/NoLeafEdgesWithSplitValues/DT_20.graph
    processing threshold 2 for forests/wine-quality/NoLeafEdgesWithSplitValues/ET_20.graph
    processing threshold 2 for forests/wine-quality/NoLeafEdgesWithSplitValues/RF_20.graph


### Next Steps

The results of this mining process are plotted in the python3 notebook 'Results for Frequent Rooted Subtrees - With Split Values in Labels.ipynb'.
