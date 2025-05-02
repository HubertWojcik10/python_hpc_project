#!/bin/sh
#BSUB -q c02613
#BSUB -J t12
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 03:00
#BSUB -o batch_output/t12_%J.out
#BSUB -e batch_output/t12_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python simulate.py 4571