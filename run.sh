#!/bin/bash
dir="/mnt/netapp2/Store_uni/home/ulc/co/jlb/redes-tf/data/"
count=0
for i in $dir/*
do
        count=$((count+1))
done

for i in `seq count`
do
        sbatch launch.sh $i
done