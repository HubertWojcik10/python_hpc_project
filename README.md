# How to copy the data

- login to the HPC 
- get out of the login node (linuxsh)
- clone the repository 
- copy the data using the following command:
__cp -r  /dtu/projects/02613_2025/data/modified_swiss_dwellings/ data__


# Repo Structure 
Tasks:
## 1-4: initial_performance_analysis
1. Familiarize yourself with the data. Load and visualize the input data for a few floorplans using a
seperate Python script, Jupyter notebook or your preferred tool. --> __eda.ipynb__
2. Familiarize yourself with the provided script. Run and time the reference implementation for asmall subset of floorplans (e.g., 10 - 20). How long do you estimate it would take to process allthe floorplans? Perform the timing as a batch job so you get reliable result? --> __time_simulation.py__