#!/bin/bash

for i in `ls /mnt/netapp2/Store_uni/home/ulc/co/jlb/redes-tf/data/*.csv`
do
	echo now is $i
	cat $i
done