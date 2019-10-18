#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) compile target (arm or intel or ppc)"
  exit 1
fi


echo "path,filename,mean,variance,min,max,size" > results_$1.csv

for d in ./*/; do
	if [ "$d" != "./__pycache__/" ]; then
		#echo $d/cpp/$1/results.csv
	
		if [ -f $d/cpp/$1/results.csv ] ; then
			rm $d/cpp/$1/results.csv
		fi

		echo "Profiling $d"
		./run.sh $d $1
		cat $d/cpp/$1/results.csv >> results_$1.csv
	fi
done