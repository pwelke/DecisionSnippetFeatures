#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) sub-folder"
  exit 1
fi

if [ "$#" -lt 2 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

cd $1/cpp/$2

for d in ./*/; do
	cd $d
	make
	cd ..
done