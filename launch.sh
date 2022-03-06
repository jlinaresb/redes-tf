#!/bin/bash

#SBATCH -p shared
#SBATCH -c 16
#SBATCH --qos shared_short

compute --mem 10
module load cesga/2018 gcc/6.4.0 tensorflow/2.2.1-python-3.8.1 seaborn/0.10.0-python-3.8.1
python pipeline.py
