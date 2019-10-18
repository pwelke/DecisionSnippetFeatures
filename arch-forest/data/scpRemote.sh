#/bin/bash
if [ "$#" -lt 1 ]
then
  echo "Please give a (valid) target (arm or intel or ppc) to copy"
  exit 1
fi

if [ "$#" -lt 2 ]
then
  echo "Please give a (valid) "account@ip address:targeted folder" of your remote device"
  exit 1
fi

mkdir tmp
find . -type f | grep "test.csv" | tar -T - -c | tar -xpC tmp
find . -perm -111 -type f | grep $1 | tar -T - -c | tar -xpC tmp
cp run.sh tmp/.
cp run_all.sh tmp/.
scp -r tmp $2
rm -r tmp
