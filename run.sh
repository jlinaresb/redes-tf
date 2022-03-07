#!/bin/bash

files=`ls /mnt/netapp2/Store_uni/home/ulc/co/jlb/redes-tf/data/*.csv`
for i in $files
do
	echo -f $i
done