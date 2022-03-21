#!/bin/bash

#SBATCH -p shared
#SBATCH -c 16
#SBATCH --qos shared_short
#SBATCH --mem 32GB
#SBATCH -t 01:00:00

module load cesga/2018 gcc/6.4.0 pandas/1.0.0-python-3.8.1 scipy/1.4.1-python-3.8.1 tensorflow/2.2.1-python-3.8.1

python pipeline_classif.py
