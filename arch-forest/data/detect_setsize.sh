#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

count=`ls -1 ./wearable-body-postures/text/*.json 2>/dev/null | wc -l`

if [ $count == 0 ];
then
	echo "Training forest"
	cd wearable-body-postures
	./trainForest.py
	cd ..
else
	echo "Trained forest found - If you want to train a new one, please delete all old files"
fi

echo "setsize,config,Mean-StandardIfTree,Var-StandardIfTree,Mean-OptimizedIfTree, Var-OptimizedIfTree,Mean-StandardNativeTree,Var-StandardNativeTree,Mean-OptimizedNativeTree,Var-OptimizedNativeTree" > results.txt
for i in 2 3 4 5 6 7 8 9 10 11 12 13 14
do
	echo -e "\tGenerating Code for setsize $i"
	./generateCode.py wearable-body-postures $1 $i
	echo -e "\tCompiling code for setsize $i"
	./compile.sh wearable-body-postures $1
	echo -e "\tProfiling code for setsize $i"

	if [ "$#" -lt 2 ] || [ "$2" != "dry" ]; 
	then
		echo "Profiling forest"
		./run.sh wearable-body-postures $1 | sed -e "s/^/$i,/" >> results.txt
	else
		echo "This is a dry run. Not getting results"
		for d in ./wearable-body-postures/cpp/$1/*/; do
			mv ./wearable-body-postures/cpp/$1/$d ./wearable-body-postures/cpp/$1/$d_$i
		done
	fi
done
