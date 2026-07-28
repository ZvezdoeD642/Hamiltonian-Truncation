import math
import numpy as np

#creating matrix
a, b, c, d = 4, 1, 2, 3
matrix = np.array([[a, c], [b, d]])

#finding eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)