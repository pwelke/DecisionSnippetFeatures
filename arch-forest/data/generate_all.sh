#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) compile target (arm or intel)"
  exit 1
fi

for d in ./*/; do
	if [ "$d" != "./__pycache__/" ]; then
        echo "Generating $d for $1"
        ./generateCode.py $d $1
    fi
done
