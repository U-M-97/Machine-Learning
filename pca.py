import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(10, 50, 100).reshape(20, 5)
mean = x - np.mean(x, axis = 0)
covar = np.cov(mean, rowvar=False)
eigen_values, eigen_vectors = np.linalg.eigh(covar)
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalue = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:, sorted_index]
n_components = 2
eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
x_reduced = np.dot(eigenvector_subset.transpose(), mean.transpose()).transpose()

# Plot original data, centered data, covariance matrix, eigenvalues, eigenvectors, and reduced data
plt.figure(figsize=(18, 12))

# Original Data
plt.subplot(2, 3, 1)
plt.title('Original Data')
plt.scatter(x[:, 0], x[:, 1], c='b', label='Original Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Centered Data
plt.subplot(2, 3, 2)
plt.title('Centered Data')
plt.scatter(mean[:, 0], mean[:, 1], c='r', label='Centered Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Covariance Matrix
plt.subplot(2, 3, 3)
plt.title('Covariance Matrix')
plt.imshow(covar, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xlabel('Variable Index')
plt.ylabel('Variable Index')

# Plot Eigenvalues and Eigenvectors
plt.subplot(2, 3, 4)
plt.title('Eigenvalues and Eigenvectors')
plt.plot(eigen_values, marker='o', linestyle='-', label='Unsorted Eigenvalues')
for i in range(eigen_vectors.shape[1]):
    plt.quiver(0, 0, eigen_vectors[0, i], eigen_vectors[1, i], angles='xy', scale_units='xy', scale=1, color='b')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Plot sorted Eigenvalues and Eigenvectors
plt.subplot(2, 3, 5)
plt.title('Sorted Eigenvalues and Eigenvectors')
plt.plot(sorted_eigenvalue, marker='o', linestyle='-', label='Sorted Eigenvalues')
for i in range(sorted_eigenvectors.shape[1]):
    plt.quiver(0, 0, sorted_eigenvectors[0, i], sorted_eigenvectors[1, i], angles='xy', scale_units='xy', scale=1, color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Plot subset of Eigenvectors
plt.subplot(2, 3, 6)
plt.title('Subset of Eigenvectors')
plt.plot(eigenvector_subset[0], eigenvector_subset[1], marker='o', linestyle='-', label='Subset of Eigenvectors')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Plot Reduced Data
plt.subplot(2, 3, 6)
plt.title('Reduced Data')
plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c='g', label='Reduced Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

plt.tight_layout()
plt.show()
