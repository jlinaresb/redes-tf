#!/bin/bash

#SBATCH -p cola-corta
#SBATCH -t 05:00:00
#SBATCH --mail-user=joselb1993@gmail.com
#SBATCH --mail-type=BEGIN,END

module load cesga/2018 gcc/6.4.0 R/3.6.3
Rscript /home/ulc/co/jlb/git/redes-tf/deconvolution/run_cibersort.r