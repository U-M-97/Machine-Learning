import numpy as np
import matplotlib.pyplot as plt

def preprocess_data(x):
    x_normalized = (x - np.mean(x, axis = 1, keepdims=True)) / np.std(x, axis = 1, keepdims=True)
    return x_normalized

def whitening(x):
    sigma = np.cov(x)
    u, s, _ = np.linalg.svd(sigma)
    whitened = np.dot(np.dot(u / np.sqrt(s), u.T), x)
    return whitened

#formula = S = W . X
def ica(x, max_iter = 1000, tol = 1e-5, learning_rate = 1e-3):
    n_comp, n_samples = x.shape
    w = np.random.rand(n_comp, n_comp)
    for _ in range(max_iter):
        w_prev = w.copy()
        est_sources = np.dot(w, x)
        est_sources = np.mean(est_sources, axis = 1, keepdims=True)
        W_grad = (n_samples * np.eye(n_comp) + (1 - 2 * np.tanh(est_sources) ** 2) @ est_sources.T) @ w
        w_grad = np.linalg.inv(w.T)
        w += learning_rate * w_grad
        if np.linalg.norm(w - w_prev) < tol:
            break
    return w

n_samples = 1000
n_sources = 3

#Mixing Matrix Formula = X = AS
mixing_matrix = np.random.rand(n_sources, n_sources)
observed_sources = np.random.rand(n_sources, n_samples)
mixed_sources = np.dot(mixing_matrix, observed_sources)

mixed_sources_normalized = preprocess_data(mixed_sources)
whitened_sources = whitening(mixed_sources_normalized)
estimated_weights = ica(whitened_sources)

estimated_sources = np.dot(estimated_weights, whitened_sources)

plt.figure(figsize=(10, 12))

# Plot original mixed sources
for i in range(n_sources):
    plt.subplot(2, n_sources, i+1)
    plt.plot(observed_sources[i])
    plt.title(f'Original Source {i+1}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

# Plot estimated sources after ICA
for i in range(n_sources):
    plt.subplot(2, n_sources, i+1+n_sources)
    plt.plot(estimated_sources[i])
    plt.title(f'Estimated Source {i+1} After ICA')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

