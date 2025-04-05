#!/bin/bash
#BSUB -J t523                               # Job name
#BSUB -q hpc                                # Queue name
#BSUB -W 30                                 # Wall times
#BSUB -R "rusage[mem=16GB]"                 # Memory request per core
#BSUB -n 23                                 # Number of cores
#BSUB -R "span[hosts=1]"                    # Number of nodes
#BSUB -R "select[model == XeonGold6126]"    # CPU model
#BSUB -o batch_output/t5_23_%J.out          # Standard output file
#BSUB -e batch_output/t5_23_%J.err          # Standard error file

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run Python script
python simulate.py 50
