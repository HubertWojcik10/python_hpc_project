#!/bin/bash
#BSUB -J visualization_task3
#BSUB -q hpc
#BSUB -W 20
#BSUB -R "rusage[mem=2048]"
#BSUB -R "select[model==XeonGold6126]"
#BSUB -o visualization_task3_%J.out
#BSUB -e visualization_task3_%J.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# use lsinfo -m to get the CPU model name
python simulation_with_visualization.py 5