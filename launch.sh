#!/bin/bash

#SBATCH -p cola-corta
#SBATCH -c 16
#SBATCH --qos default
#SBATCH --mem 8GB
#SBATCH -t 01:00:00

module load cesga/2018 gcc/6.4.0 pandas/1.0.0-python-3.8.1 tensorflow/2.2.1-python-3.8.1

python pipeline_v2.py -f $i
