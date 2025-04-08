#!/bin/bash
#BSUB -J timed_simulation_task2_5N
#BSUB -q hpc
#BSUB -W 20
#BSUB -R "rusage[mem=2048]"
#BSUB -R "select[model==XeonGold6126]"
#BSUB -o timed_simulation_task2_5N_%J.out
#BSUB -e timed_simulation_task2_5N_%J.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# use lsinfo -m to get the CPU model name
python timed_simulation.py 5