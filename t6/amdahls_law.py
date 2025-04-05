import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

processors = list(range(1,24))
timings = [716.8819859027863, 371.9506411552429, 383.2784492969513, 263.3111882209778, 224.853346824646,
        181.4168581962, 148.65775537490845, 158.05766320228577, 122.6216676235199, 112.85656499862671,
        103.46184945106506, 104.98900508880615, 94.49129056930542, 87.00912237167358, 79.03059577941895,
        79.19486904144287, 79.05181193351746, 72.77513074874878, 73.45229291915894, 71.30902028083801,
        71.50569462776184, 72.4660234451294, 74.23082900047302]

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