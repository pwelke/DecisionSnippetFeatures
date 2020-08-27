# Reducing Distinct Branching Conditions in Decision Forests

We are following the paper 'An Algorithm for Reducing the Number of Distinct Branching Conditions in a Decision Forest' by Nakamura and Sakurada.

## Datasets

We will use the 'adult' and 'wine-quality' datasets only. The data are given as json files of the following names:


```bash
%%bash
rm forests/*/text/*pruned_with*.json
ls forests/*/text/*.json
```

    forests/adult/text/DT_10.json
    forests/adult/text/DT_15.json
    forests/adult/text/DT_1.json
    forests/adult/text/DT_20.json
    forests/adult/text/DT_5.json
    forests/adult/text/ET_10.json
    forests/adult/text/ET_15.json
    forests/adult/text/ET_1.json
    forests/adult/text/ET_20.json
    forests/adult/text/ET_5.json
    forests/adult/text/RF_10.json
    forests/adult/text/RF_15.json
    forests/adult/text/RF_1.json
    forests/adult/text/RF_20.json
    forests/adult/text/RF_5.json
    forests/wine-quality/text/DT_10.json
    forests/wine-quality/text/DT_15.json
    forests/wine-quality/text/DT_1.json
    forests/wine-quality/text/DT_20.json
    forests/wine-quality/text/DT_5.json
    forests/wine-quality/text/ET_10.json
    forests/wine-quality/text/ET_15.json
    forests/wine-quality/text/ET_1.json
    forests/wine-quality/text/ET_20.json
    forests/wine-quality/text/ET_5.json
    forests/wine-quality/text/RF_10.json
    forests/wine-quality/text/RF_15.json
    forests/wine-quality/text/RF_1.json
    forests/wine-quality/text/RF_20.json
    forests/wine-quality/text/RF_5.json


We now prune these decision forests with $\sigma = 0.1$. <br>
TODO: do the pruning for $\sigma \in \{0.0, 0.1, 0.2, 0.3 \}$.

This took about 5 to 6 minutes for my laptop.


```bash
%%bash
for sigma in 0.0 0.1 0.2 0.3; do (
    for dataset in adult wine-quality; do (
        for f in forests/${dataset}/text/*.json; do
            echo ${f} '->' `basename ${f} .json`_pruned_with_sigma_${sigma}.json
            ./Pruning/pruning.py ${f} forests/${dataset}/FeatureVectors.dat ${sigma}
        done ) & #The '&' character here parallelizes it on 8 threads
    done ) &
done
```

    forests/adult/text/DT_10.json -> DT_10_pruned_with_sigma_0.0.json
    forests/wine-quality/text/DT_10.json -> DT_10_pruned_with_sigma_0.1.json
    forests/wine-quality/text/DT_10.json -> DT_10_pruned_with_sigma_0.0.json
    forests/adult/text/DT_10.json -> DT_10_pruned_with_sigma_0.3.json
    forests/wine-quality/text/DT_10.json -> DT_10_pruned_with_sigma_0.3.json
    forests/adult/text/DT_10.json -> DT_10_pruned_with_sigma_0.1.json
    forests/adult/text/DT_10.json -> DT_10_pruned_with_sigma_0.2.json
    forests/wine-quality/text/DT_10.json -> DT_10_pruned_with_sigma_0.2.json
    forests/wine-quality/text/DT_15.json -> DT_15_pruned_with_sigma_0.3.json
    forests/wine-quality/text/DT_15.json -> DT_15_pruned_with_sigma_0.0.json
    forests/wine-quality/text/DT_15.json -> DT_15_pruned_with_sigma_0.2.json
    forests/wine-quality/text/DT_15.json -> DT_15_pruned_with_sigma_0.1.json
    forests/adult/text/DT_15.json -> DT_15_pruned_with_sigma_0.0.json
    forests/adult/text/DT_15.json -> DT_15_pruned_with_sigma_0.2.json
    forests/adult/text/DT_15.json -> DT_15_pruned_with_sigma_0.1.json
    forests/adult/text/DT_15.json -> DT_15_pruned_with_sigma_0.3.json
    forests/wine-quality/text/DT_1.json -> DT_1_pruned_with_sigma_0.3.json
    forests/wine-quality/text/DT_1.json -> DT_1_pruned_with_sigma_0.0.json
    forests/wine-quality/text/DT_1.json -> DT_1_pruned_with_sigma_0.1.json
    forests/wine-quality/text/DT_1.json -> DT_1_pruned_with_sigma_0.2.json
    forests/wine-quality/text/DT_20.json -> DT_20_pruned_with_sigma_0.3.json
    forests/wine-quality/text/DT_20.json -> DT_20_pruned_with_sigma_0.0.json
    forests/wine-quality/text/DT_20.json -> DT_20_pruned_with_sigma_0.1.json
    forests/wine-quality/text/DT_20.json -> DT_20_pruned_with_sigma_0.2.json
    forests/adult/text/DT_1.json -> DT_1_pruned_with_sigma_0.2.json
    forests/wine-quality/text/DT_5.json -> DT_5_pruned_with_sigma_0.3.json
    forests/adult/text/DT_1.json -> DT_1_pruned_with_sigma_0.0.json
    forests/wine-quality/text/DT_5.json -> DT_5_pruned_with_sigma_0.1.json
    forests/adult/text/DT_1.json -> DT_1_pruned_with_sigma_0.3.json
    forests/wine-quality/text/DT_5.json -> DT_5_pruned_with_sigma_0.0.json
    forests/adult/text/DT_1.json -> DT_1_pruned_with_sigma_0.1.json
    forests/wine-quality/text/DT_5.json -> DT_5_pruned_with_sigma_0.2.json
    forests/wine-quality/text/ET_10.json -> ET_10_pruned_with_sigma_0.3.json
    forests/adult/text/DT_20.json -> DT_20_pruned_with_sigma_0.2.json
    forests/adult/text/DT_20.json -> DT_20_pruned_with_sigma_0.0.json
    forests/wine-quality/text/ET_10.json -> ET_10_pruned_with_sigma_0.0.json
    forests/adult/text/DT_20.json -> DT_20_pruned_with_sigma_0.1.json
    forests/wine-quality/text/ET_10.json -> ET_10_pruned_with_sigma_0.1.json
    forests/adult/text/DT_20.json -> DT_20_pruned_with_sigma_0.3.json
    forests/wine-quality/text/ET_10.json -> ET_10_pruned_with_sigma_0.2.json
    forests/adult/text/DT_5.json -> DT_5_pruned_with_sigma_0.2.json
    forests/adult/text/DT_5.json -> DT_5_pruned_with_sigma_0.0.json
    forests/adult/text/DT_5.json -> DT_5_pruned_with_sigma_0.3.json
    forests/adult/text/DT_5.json -> DT_5_pruned_with_sigma_0.1.json
    forests/adult/text/ET_10.json -> ET_10_pruned_with_sigma_0.0.json
    forests/adult/text/ET_10.json -> ET_10_pruned_with_sigma_0.2.json
    forests/adult/text/ET_10.json -> ET_10_pruned_with_sigma_0.3.json
    forests/adult/text/ET_10.json -> ET_10_pruned_with_sigma_0.1.json
    forests/wine-quality/text/ET_15.json -> ET_15_pruned_with_sigma_0.3.json
    forests/wine-quality/text/ET_15.json -> ET_15_pruned_with_sigma_0.0.json
    forests/wine-quality/text/ET_15.json -> ET_15_pruned_with_sigma_0.1.json
    forests/wine-quality/text/ET_15.json -> ET_15_pruned_with_sigma_0.2.json
    forests/wine-quality/text/ET_1.json -> ET_1_pruned_with_sigma_0.3.json
    forests/wine-quality/text/ET_1.json -> ET_1_pruned_with_sigma_0.2.json
    forests/wine-quality/text/ET_1.json -> ET_1_pruned_with_sigma_0.0.json
    forests/wine-quality/text/ET_1.json -> ET_1_pruned_with_sigma_0.1.json
    forests/wine-quality/text/ET_20.json -> ET_20_pruned_with_sigma_0.3.json
    forests/wine-quality/text/ET_20.json -> ET_20_pruned_with_sigma_0.2.json
    forests/wine-quality/text/ET_20.json -> ET_20_pruned_with_sigma_0.0.json
    forests/wine-quality/text/ET_20.json -> ET_20_pruned_with_sigma_0.1.json
    forests/wine-quality/text/ET_5.json -> ET_5_pruned_with_sigma_0.3.json
    forests/wine-quality/text/ET_5.json -> ET_5_pruned_with_sigma_0.2.json
    forests/wine-quality/text/ET_5.json -> ET_5_pruned_with_sigma_0.1.json
    forests/wine-quality/text/ET_5.json -> ET_5_pruned_with_sigma_0.0.json
    forests/adult/text/ET_15.json -> ET_15_pruned_with_sigma_0.2.json
    forests/adult/text/ET_15.json -> ET_15_pruned_with_sigma_0.3.json
    forests/adult/text/ET_15.json -> ET_15_pruned_with_sigma_0.1.json
    forests/adult/text/ET_15.json -> ET_15_pruned_with_sigma_0.0.json
    forests/wine-quality/text/RF_10.json -> RF_10_pruned_with_sigma_0.3.json
    forests/wine-quality/text/RF_10.json -> RF_10_pruned_with_sigma_0.2.json
    forests/wine-quality/text/RF_10.json -> RF_10_pruned_with_sigma_0.1.json
    forests/wine-quality/text/RF_10.json -> RF_10_pruned_with_sigma_0.0.json
    forests/wine-quality/text/RF_15.json -> RF_15_pruned_with_sigma_0.3.json
    forests/wine-quality/text/RF_15.json -> RF_15_pruned_with_sigma_0.2.json
    forests/wine-quality/text/RF_15.json -> RF_15_pruned_with_sigma_0.1.json
    forests/wine-quality/text/RF_15.json -> RF_15_pruned_with_sigma_0.0.json
    forests/wine-quality/text/RF_1.json -> RF_1_pruned_with_sigma_0.2.json
    forests/wine-quality/text/RF_1.json -> RF_1_pruned_with_sigma_0.3.json
    forests/wine-quality/text/RF_1.json -> RF_1_pruned_with_sigma_0.1.json
    forests/wine-quality/text/RF_1.json -> RF_1_pruned_with_sigma_0.0.json
    forests/wine-quality/text/RF_20.json -> RF_20_pruned_with_sigma_0.2.json
    forests/wine-quality/text/RF_20.json -> RF_20_pruned_with_sigma_0.3.json
    forests/wine-quality/text/RF_20.json -> RF_20_pruned_with_sigma_0.0.json
    forests/wine-quality/text/RF_20.json -> RF_20_pruned_with_sigma_0.1.json
    forests/wine-quality/text/RF_5.json -> RF_5_pruned_with_sigma_0.0.json
    forests/wine-quality/text/RF_5.json -> RF_5_pruned_with_sigma_0.3.json
    forests/wine-quality/text/RF_5.json -> RF_5_pruned_with_sigma_0.2.json
    forests/wine-quality/text/RF_5.json -> RF_5_pruned_with_sigma_0.1.json
    forests/adult/text/ET_1.json -> ET_1_pruned_with_sigma_0.2.json
    forests/adult/text/ET_1.json -> ET_1_pruned_with_sigma_0.1.json
    forests/adult/text/ET_1.json -> ET_1_pruned_with_sigma_0.3.json
    forests/adult/text/ET_1.json -> ET_1_pruned_with_sigma_0.0.json
    forests/adult/text/ET_20.json -> ET_20_pruned_with_sigma_0.2.json
    forests/adult/text/ET_20.json -> ET_20_pruned_with_sigma_0.1.json
    forests/adult/text/ET_20.json -> ET_20_pruned_with_sigma_0.3.json
    forests/adult/text/ET_20.json -> ET_20_pruned_with_sigma_0.0.json
    forests/adult/text/ET_5.json -> ET_5_pruned_with_sigma_0.1.json
    forests/adult/text/ET_5.json -> ET_5_pruned_with_sigma_0.2.json
    forests/adult/text/ET_5.json -> ET_5_pruned_with_sigma_0.3.json
    forests/adult/text/ET_5.json -> ET_5_pruned_with_sigma_0.0.json
    forests/adult/text/RF_10.json -> RF_10_pruned_with_sigma_0.1.json
    forests/adult/text/RF_10.json -> RF_10_pruned_with_sigma_0.2.json
    forests/adult/text/RF_10.json -> RF_10_pruned_with_sigma_0.3.json
    forests/adult/text/RF_10.json -> RF_10_pruned_with_sigma_0.0.json
    forests/adult/text/RF_15.json -> RF_15_pruned_with_sigma_0.1.json
    forests/adult/text/RF_15.json -> RF_15_pruned_with_sigma_0.2.json
    forests/adult/text/RF_15.json -> RF_15_pruned_with_sigma_0.3.json
    forests/adult/text/RF_15.json -> RF_15_pruned_with_sigma_0.0.json
    forests/adult/text/RF_1.json -> RF_1_pruned_with_sigma_0.3.json
    forests/adult/text/RF_1.json -> RF_1_pruned_with_sigma_0.2.json
    forests/adult/text/RF_1.json -> RF_1_pruned_with_sigma_0.1.json
    forests/adult/text/RF_1.json -> RF_1_pruned_with_sigma_0.0.json
    forests/adult/text/RF_20.json -> RF_20_pruned_with_sigma_0.3.json
    forests/adult/text/RF_20.json -> RF_20_pruned_with_sigma_0.2.json
    forests/adult/text/RF_20.json -> RF_20_pruned_with_sigma_0.1.json
    forests/adult/text/RF_20.json -> RF_20_pruned_with_sigma_0.0.json
    forests/adult/text/RF_5.json -> RF_5_pruned_with_sigma_0.1.json
    forests/adult/text/RF_5.json -> RF_5_pruned_with_sigma_0.2.json
    forests/adult/text/RF_5.json -> RF_5_pruned_with_sigma_0.3.json
    forests/adult/text/RF_5.json -> RF_5_pruned_with_sigma_0.0.json



```python
ls forests/*/text/*.json
```

    forests/adult/text/DT_10.json
    forests/adult/text/DT_10_pruned_with_sigma_0_0.json
    forests/adult/text/DT_10_pruned_with_sigma_0_1.json
    forests/adult/text/DT_10_pruned_with_sigma_0_2.json
    forests/adult/text/DT_10_pruned_with_sigma_0_3.json
    forests/adult/text/DT_15.json
    forests/adult/text/DT_15_pruned_with_sigma_0_0.json
    forests/adult/text/DT_15_pruned_with_sigma_0_1.json
    forests/adult/text/DT_15_pruned_with_sigma_0_2.json
    forests/adult/text/DT_15_pruned_with_sigma_0_3.json
    forests/adult/text/DT_1.json
    forests/adult/text/DT_1_pruned_with_sigma_0_0.json
    forests/adult/text/DT_1_pruned_with_sigma_0_1.json
    forests/adult/text/DT_1_pruned_with_sigma_0_2.json
    forests/adult/text/DT_1_pruned_with_sigma_0_3.json
    forests/adult/text/DT_20.json
    forests/adult/text/DT_20_pruned_with_sigma_0_0.json
    forests/adult/text/DT_20_pruned_with_sigma_0_1.json
    forests/adult/text/DT_20_pruned_with_sigma_0_2.json
    forests/adult/text/DT_20_pruned_with_sigma_0_3.json
    forests/adult/text/DT_5.json
    forests/adult/text/DT_5_pruned_with_sigma_0_0.json
    forests/adult/text/DT_5_pruned_with_sigma_0_1.json
    forests/adult/text/DT_5_pruned_with_sigma_0_2.json
    forests/adult/text/DT_5_pruned_with_sigma_0_3.json
    forests/adult/text/ET_10.json
    forests/adult/text/ET_10_pruned_with_sigma_0_0.json
    forests/adult/text/ET_10_pruned_with_sigma_0_1.json
    forests/adult/text/ET_10_pruned_with_sigma_0_2.json
    forests/adult/text/ET_10_pruned_with_sigma_0_3.json
    forests/adult/text/ET_15.json
    forests/adult/text/ET_15_pruned_with_sigma_0_0.json
    forests/adult/text/ET_15_pruned_with_sigma_0_1.json
    forests/adult/text/ET_15_pruned_with_sigma_0_2.json
    forests/adult/text/ET_15_pruned_with_sigma_0_3.json
    forests/adult/text/ET_1.json
    forests/adult/text/ET_1_pruned_with_sigma_0_0.json
    forests/adult/text/ET_1_pruned_with_sigma_0_1.json
    forests/adult/text/ET_1_pruned_with_sigma_0_2.json
    forests/adult/text/ET_1_pruned_with_sigma_0_3.json
    forests/adult/text/ET_20.json
    forests/adult/text/ET_20_pruned_with_sigma_0_0.json
    forests/adult/text/ET_20_pruned_with_sigma_0_1.json
    forests/adult/text/ET_20_pruned_with_sigma_0_2.json
    forests/adult/text/ET_20_pruned_with_sigma_0_3.json
    forests/adult/text/ET_5.json
    forests/adult/text/ET_5_pruned_with_sigma_0_0.json
    forests/adult/text/ET_5_pruned_with_sigma_0_1.json
    forests/adult/text/ET_5_pruned_with_sigma_0_2.json
    forests/adult/text/ET_5_pruned_with_sigma_0_3.json
    forests/adult/text/RF_10.json
    forests/adult/text/RF_10_pruned_with_sigma_0_0.json
    forests/adult/text/RF_10_pruned_with_sigma_0_1.json
    forests/adult/text/RF_10_pruned_with_sigma_0_2.json
    forests/adult/text/RF_10_pruned_with_sigma_0_3.json
    forests/adult/text/RF_15.json
    forests/adult/text/RF_15_pruned_with_sigma_0_0.json
    forests/adult/text/RF_15_pruned_with_sigma_0_1.json
    forests/adult/text/RF_15_pruned_with_sigma_0_2.json
    forests/adult/text/RF_15_pruned_with_sigma_0_3.json
    forests/adult/text/RF_1.json
    forests/adult/text/RF_1_pruned_with_sigma_0_0.json
    forests/adult/text/RF_1_pruned_with_sigma_0_1.json
    forests/adult/text/RF_1_pruned_with_sigma_0_2.json
    forests/adult/text/RF_1_pruned_with_sigma_0_3.json
    forests/adult/text/RF_20.json
    forests/adult/text/RF_20_pruned_with_sigma_0_0.json
    forests/adult/text/RF_20_pruned_with_sigma_0_1.json
    forests/adult/text/RF_20_pruned_with_sigma_0_2.json
    forests/adult/text/RF_20_pruned_with_sigma_0_3.json
    forests/adult/text/RF_5.json
    forests/adult/text/RF_5_pruned_with_sigma_0_0.json
    forests/adult/text/RF_5_pruned_with_sigma_0_1.json
    forests/adult/text/RF_5_pruned_with_sigma_0_2.json
    forests/adult/text/RF_5_pruned_with_sigma_0_3.json
    forests/wine-quality/text/DT_10.json
    forests/wine-quality/text/DT_10_pruned_with_sigma_0_0.json
    forests/wine-quality/text/DT_10_pruned_with_sigma_0_1.json
    forests/wine-quality/text/DT_10_pruned_with_sigma_0_2.json
    forests/wine-quality/text/DT_10_pruned_with_sigma_0_3.json
    forests/wine-quality/text/DT_15.json
    forests/wine-quality/text/DT_15_pruned_with_sigma_0_0.json
    forests/wine-quality/text/DT_15_pruned_with_sigma_0_1.json
    forests/wine-quality/text/DT_15_pruned_with_sigma_0_2.json
    forests/wine-quality/text/DT_15_pruned_with_sigma_0_3.json
    forests/wine-quality/text/DT_1.json
    forests/wine-quality/text/DT_1_pruned_with_sigma_0_0.json
    forests/wine-quality/text/DT_1_pruned_with_sigma_0_1.json
    forests/wine-quality/text/DT_1_pruned_with_sigma_0_2.json
    forests/wine-quality/text/DT_1_pruned_with_sigma_0_3.json
    forests/wine-quality/text/DT_20.json
    forests/wine-quality/text/DT_20_pruned_with_sigma_0_0.json
    forests/wine-quality/text/DT_20_pruned_with_sigma_0_1.json
    forests/wine-quality/text/DT_20_pruned_with_sigma_0_2.json
    forests/wine-quality/text/DT_20_pruned_with_sigma_0_3.json
    forests/wine-quality/text/DT_5.json
    forests/wine-quality/text/DT_5_pruned_with_sigma_0_0.json
    forests/wine-quality/text/DT_5_pruned_with_sigma_0_1.json
    forests/wine-quality/text/DT_5_pruned_with_sigma_0_2.json
    forests/wine-quality/text/DT_5_pruned_with_sigma_0_3.json
    forests/wine-quality/text/ET_10.json
    forests/wine-quality/text/ET_10_pruned_with_sigma_0_0.json
    forests/wine-quality/text/ET_10_pruned_with_sigma_0_1.json
    forests/wine-quality/text/ET_10_pruned_with_sigma_0_2.json
    forests/wine-quality/text/ET_10_pruned_with_sigma_0_3.json
    forests/wine-quality/text/ET_15.json
    forests/wine-quality/text/ET_15_pruned_with_sigma_0_0.json
    forests/wine-quality/text/ET_15_pruned_with_sigma_0_1.json
    forests/wine-quality/text/ET_15_pruned_with_sigma_0_2.json
    forests/wine-quality/text/ET_15_pruned_with_sigma_0_3.json
    forests/wine-quality/text/ET_1.json
    forests/wine-quality/text/ET_1_pruned_with_sigma_0_0.json
    forests/wine-quality/text/ET_1_pruned_with_sigma_0_1.json
    forests/wine-quality/text/ET_1_pruned_with_sigma_0_2.json
    forests/wine-quality/text/ET_1_pruned_with_sigma_0_3.json
    forests/wine-quality/text/ET_20.json
    forests/wine-quality/text/ET_20_pruned_with_sigma_0_0.json
    forests/wine-quality/text/ET_20_pruned_with_sigma_0_1.json
    forests/wine-quality/text/ET_20_pruned_with_sigma_0_2.json
    forests/wine-quality/text/ET_20_pruned_with_sigma_0_3.json
    forests/wine-quality/text/ET_5.json
    forests/wine-quality/text/ET_5_pruned_with_sigma_0_0.json
    forests/wine-quality/text/ET_5_pruned_with_sigma_0_1.json
    forests/wine-quality/text/ET_5_pruned_with_sigma_0_2.json
    forests/wine-quality/text/ET_5_pruned_with_sigma_0_3.json
    forests/wine-quality/text/RF_10.json
    forests/wine-quality/text/RF_10_pruned_with_sigma_0_0.json
    forests/wine-quality/text/RF_10_pruned_with_sigma_0_1.json
    forests/wine-quality/text/RF_10_pruned_with_sigma_0_2.json
    forests/wine-quality/text/RF_10_pruned_with_sigma_0_3.json
    forests/wine-quality/text/RF_15.json
    forests/wine-quality/text/RF_15_pruned_with_sigma_0_0.json
    forests/wine-quality/text/RF_15_pruned_with_sigma_0_1.json
    forests/wine-quality/text/RF_15_pruned_with_sigma_0_2.json
    forests/wine-quality/text/RF_15_pruned_with_sigma_0_3.json
    forests/wine-quality/text/RF_1.json
    forests/wine-quality/text/RF_1_pruned_with_sigma_0_0.json
    forests/wine-quality/text/RF_1_pruned_with_sigma_0_1.json
    forests/wine-quality/text/RF_1_pruned_with_sigma_0_2.json
    forests/wine-quality/text/RF_1_pruned_with_sigma_0_3.json
    forests/wine-quality/text/RF_20.json
    forests/wine-quality/text/RF_20_pruned_with_sigma_0_0.json
    forests/wine-quality/text/RF_20_pruned_with_sigma_0_1.json
    forests/wine-quality/text/RF_20_pruned_with_sigma_0_2.json
    forests/wine-quality/text/RF_20_pruned_with_sigma_0_3.json
    forests/wine-quality/text/RF_5.json
    forests/wine-quality/text/RF_5_pruned_with_sigma_0_0.json
    forests/wine-quality/text/RF_5_pruned_with_sigma_0_1.json
    forests/wine-quality/text/RF_5_pruned_with_sigma_0_2.json
    forests/wine-quality/text/RF_5_pruned_with_sigma_0_3.json



```python

```
