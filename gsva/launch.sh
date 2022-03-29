#!/bin/bash

#SBATCH -p shared
#SBATCH --qos shared_short
#SBATCH --mem 128GB
#SBATCH -t 10:00:00

module load gcc/6.4.0 R/3.6.3
Rscript ~/git/redes-tf/gsva/gsva_msigDB.r