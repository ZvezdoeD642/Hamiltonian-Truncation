import matplotlib.pyplot as plt
import numpy as np

sizes = np.array([50, 100, 200, 500, 1000])
times = np.array([0.001, 0.008, 0.064, 1.0, 8.0])  # example runtimes

# Fit log(T) vs log(N)
log_N = np.log(sizes)
log_T = np.log(times)
slope, intercept = np.polyfit(log_N, log_T, 1)

print(f"Empirical Time Complexity: O(N^{slope:.2f})")