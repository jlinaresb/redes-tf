#!/bin/bash

for i in `seq 4`
do
        sbatch launch.sh $i
done