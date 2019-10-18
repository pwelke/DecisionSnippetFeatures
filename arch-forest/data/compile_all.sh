#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) compile target (arm or intel or ppc)"
  exit 1
fi

for d in */; do
	if [ "$d" != "./__pycache__/" ]; then
		echo "Compiling $d"
		./compile.sh $d $1
	fi
done
