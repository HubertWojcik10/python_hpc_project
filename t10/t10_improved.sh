#!/bin/sh
#BSUB -q c02613
#BSUB -J t10_5N_impr
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o t10_5N_impr_%J.out
#BSUB -e t10_5N_impr_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

nsys profile -o t10_5N_impr.txt python simulate_cupy_improved.py 5