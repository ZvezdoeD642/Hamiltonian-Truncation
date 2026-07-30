import math
import random
import numpy as np
import matplotlib as plt
import time

total_time = 0
#running the code to find eigenvectors 100 times and measuring the time taken

for i in range(100):
    start_time = time.time()
    #creating 1000x1000 matrix with random values
    matrix = np.random.rand(1000, 1000)

    #finding eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    time_taken = time.time() - start_time
    total_time += time_taken

#finding average time
average_time = total_time / 100

print("Average time taken:", average_time, "seconds")

print("Total time taken:", total_time, "seconds")