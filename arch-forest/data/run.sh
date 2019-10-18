#/bin/bash

if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) sub-folder"
  exit 1
fi

if [ "$#" -lt 2 ]
then
  echo "Please give a (valid) compile target (arm or intel or ppc)"
  exit 1
fi

#echo "path,filename,treedepth,mean,variance,min,max,size"

for d in $(find ./$1/cpp/$2/*/ -executable -type f); do
	# echo $d
	cd $(dirname $d)
	bname=$(basename $d)
	# echo $bname
	# echo $(basename $(dirname $d))
	measurments="$d,$bname,$(basename $(dirname $d)),$(./$bname),$(stat --printf="%s" $bname)"
	cd ..
	echo $measurments >> results.csv
	cd ../../../
done
