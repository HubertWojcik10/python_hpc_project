#!/bin/sh
#BSUB -q c02613
#BSUB -J t8m_100N
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o t8m_100N_%J.out
#BSUB -e t8m_100N_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python simulate_with_memmap.py 100