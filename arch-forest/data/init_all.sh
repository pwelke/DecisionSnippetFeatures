#/bin/bash

for d in */; do
	cd $d
	echo "Preparing $d"
	./init.sh 
	cd .
	cd ..
done
