#!/bin/bash

search_dir=/mnt/netapp2/Store_uni/home/ulc/co/jlb/redes-tf/data/
for i in "$search_dir"/*
do
	sbatch launch.sh $i
done