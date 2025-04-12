#!/bin/bash
#BSUB -J t7_100N
#BSUB -q hpc
#BSUB -W 20
#BSUB -R "rusage[mem=2048]"
#BSUB -R "select[model==XeonGold6126]"
#BSUB -o t7_100N_5N_%J.out
#BSUB -e t7_100N_%J.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"

python simulate.py 1000