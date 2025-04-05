import matplotlib.pyplot as plt

processors = list(range(1,24))
timings = [716.8819859027863, 356.02556586265564, 264.2099804878235, 230.96657729148865, 218.68071007728577, 
        197.54756450653076, 172.80523228645325, 166.37991786003113, 137.60053181648254, 127.94726610183716,
        123.38340067863464, 122.61980843544006, 88.22018766403198, 88.02406883239746, 88.17843461036682,
        87.32292914390564, 76.65184187889099, 75.3033754825592, 74.041996717453, 75.28804087638855,
        74.31268334388733, 72.09457015991211, 73.86306071281433]

T1 = timings[0]
speed_ups = [T1 / T for T in timings]

# Plotting the speed up against the number of processors used 
plt.figure(figsize=(8, 5))
plt.plot(processors, speed_ups, marker='o', linestyle='-', color='indianred')
plt.xlabel("Number of processors")
plt.ylabel("Speed-up")
plt.title("Speed-up vs. Number of processors")
plt.grid(alpha=0.4)
plt.show()

plt.savefig("speedup_processors.png", dpi=300, bbox_inches='tight')
