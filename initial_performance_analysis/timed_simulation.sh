#!/bin/bash
#BSUB -J timed_simulation_task2
#BSUB -q hpc
#BSUB -W 2
#BSUB -R "rusage[mem=1024]"
#BSUB -R "select[model==XeonGold6126]"
#BSUB -o timed_simulation_task2_%J.out
#BSUB -e timed_simulation_task2_%J.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"

# use lsinfo -m to get the CPU model name
python timed_simulation.py 20