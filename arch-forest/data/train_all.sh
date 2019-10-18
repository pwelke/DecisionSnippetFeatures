#/bin/bash

for d in ./*/; do
	if [ "$d" != "./__pycache__/" ]; then
		cd $d
		echo "Training $d"
		./trainForest.py > train.txt
		cd ..
    fi
done
