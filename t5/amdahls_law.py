import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

processors = list(range(1,24))
timings = [716.8819859027863, 356.02556586265564, 264.2099804878235, 230.96657729148865, 218.68071007728577, 
        197.54756450653076, 172.80523228645325, 166.37991786003113, 137.60053181648254, 127.94726610183716,
        123.38340067863464, 122.61980843544006, 88.22018766403198, 88.02406883239746, 88.17843461036682,
        87.32292914390564, 76.65184187889099, 75.3033754825592, 74.041996717453, 75.28804087638855,
        74.31268334388733, 72.09457015991211, 73.86306071281433]

T1 = timings[0]
speed_ups = [T1 / T for T in timings]

# Amdahl's law formula from class
def amdahl(p, B):
    return 1 / (B + (1-B)/p)

# Fit curve to find serial fraction
params, covariance = curve_fit(amdahl, processors, speed_ups)
serial_fraction = params[0]
print(f"Serial fraction: {serial_fraction:.4f}")
print(f"Parallel fraction: {1-serial_fraction:.4f}")

# Max speed-up formula from class
max_speedup = 1/serial_fraction
print(f"Maximum speed-up: {max_speedup:.4f}")
print(f"We achieved the maximum speed-up of {max(speed_ups):.4f} using {processors[speed_ups.index(max(speed_ups))]} cores.")

N = 50
all_N = 4571
time_for_all = ((timings[speed_ups.index(max(speed_ups))])/N)*all_N

print(f"To go through all {all_N} floorplans instead of out subset of {N}, it would take {time_for_all/60:.2f} minutes.")