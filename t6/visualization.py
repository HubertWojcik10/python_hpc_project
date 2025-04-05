import matplotlib.pyplot as plt

processors = list(range(1,24))
timings = [716.8819859027863, 371.9506411552429, 383.2784492969513, 263.3111882209778, 224.853346824646,
        181.4168581962, 148.65775537490845, 158.05766320228577, 122.6216676235199, 112.85656499862671,
        103.46184945106506, 104.98900508880615, 94.49129056930542, 87.00912237167358, 79.03059577941895,
        79.19486904144287, 79.05181193351746, 72.77513074874878, 73.45229291915894, 71.30902028083801,
        71.50569462776184, 72.4660234451294, 74.23082900047302]

T1 = timings[0]
speed_ups = [T1 / T for T in timings]

# Plotting the speed up against the number of processors used 
plt.figure(figsize=(8, 5))
plt.plot(processors, timings, marker='o', linestyle='-', color='indianred')
plt.xlabel("Number of processors")
plt.ylabel("Time")
plt.title("Time vs. Number of processors")
plt.grid(alpha=0.4)
plt.show()

plt.savefig("plots/time_processors.png", dpi=300, bbox_inches='tight')
