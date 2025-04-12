#!/bin/bash
#BSUB -J t7_1000N
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "rusage[mem=4096]"
#BSUB -R "select[model==XeonGold6126]"
#BSUB -o t7_1000N_%J.out
#BSUB -e t7_1000N_%J.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python simulate.py 1000