import numpy as np

# Define matrices A (1000x10) and B (10x1000)
A = np.random.rand(1000, 10)
B = np.random.rand(10, 1000)

# Calculate the product A * B
A_dot_B = np.dot(A, B)

print(A_dot_B.shape)