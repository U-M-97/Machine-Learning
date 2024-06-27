import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

np.random.seed(0)
torch.manual_seed(0)

def initialize_params(input_size, num_neurons):
    # Xavier initialization
    weights = np.random.normal(loc=0.0, scale=np.sqrt(2 / (input_size + num_neurons)), size=(num_neurons, input_size))
    biases = np.random.normal(loc=0.0, scale=np.sqrt(2 / (input_size + num_neurons)), size=num_neurons)
    return weights, biases

def initialize_kernels(num_kernels, channels, kernel_size):
    n_inputs = channels * kernel_size * kernel_size
    n_outputs = num_kernels * kernel_size * kernel_size
    std_dev = np.sqrt(2 / (n_inputs + n_outputs))
    kernels = np.random.normal(loc=0.0, scale=std_dev, size=(num_kernels, channels, kernel_size, kernel_size))
    biases = np.random.normal(loc=0.0, scale=std_dev, size=num_kernels)
    return kernels, biases

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Compute the size of the linear layer input
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Assuming input size is 28x28
        self.fc2 = nn.Linear(128, 100)
        self.fc3 = nn.Linear(100, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        with torch.no_grad():
            # Initialize conv1
            kernels, biases = initialize_kernels(32, 1, 3)
            print(kernels[20])
            self.conv1.weight.copy_(torch.tensor(kernels, dtype=torch.float32))
            self.conv1.bias.copy_(torch.tensor(biases, dtype=torch.float32))
            
            # Initialize conv2
            kernels, biases = initialize_kernels(64, 32, 3)
            self.conv2.weight.copy_(torch.tensor(kernels, dtype=torch.float32))
            self.conv2.bias.copy_(torch.tensor(biases, dtype=torch.float32))
            
            # Initialize fc1
            weights, biases = initialize_params(64 * 5 * 5, 128)
            self.fc1.weight.copy_(torch.tensor(weights, dtype=torch.float32))
            self.fc1.bias.copy_(torch.tensor(biases, dtype=torch.float32))
            
            # Initialize fc2
            weights, biases = initialize_params(128, 100)
            self.fc2.weight.copy_(torch.tensor(weights, dtype=torch.float32))
            self.fc2.bias.copy_(torch.tensor(biases, dtype=torch.float32))
            
            # Initialize fc3
            weights, biases = initialize_params(100, num_classes)
            self.fc3.weight.copy_(torch.tensor(weights, dtype=torch.float32))
            self.fc3.bias.copy_(torch.tensor(biases, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)
        print(x)
        print(np.sum(x.detach().numpy()))
        return x

# Hook to capture the gradient
def get_grad(module, grad_input, grad_output):
    global gradients
    print(grad_input[0].shape)
    print(grad_output[0].shape)
    gradients = grad_output[0]

# Create input matrix and one-hot encoded label
input_matrix = np.random.normal(loc=0.5, scale=0.25, size=(28, 28)).astype(np.float32)
normalized_matrix = (input_matrix - np.min(input_matrix)) / (np.max(input_matrix) - np.min(input_matrix))
print(np.sum(normalized_matrix))

num_classes = 10
class_label = 5

# Convert input matrix to PyTorch tensor and add batch and channel dimensions
input_tensor = torch.tensor(normalized_matrix).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)
input_tensor.requires_grad = True  # Enable gradient calculation for the input tensor
label_tensor = torch.zeros(1, num_classes)
label_tensor[0, class_label] = 1

# Initialize the model
model = SimpleCNN(num_classes=num_classes)

# Register the hook to the second convolutional layer
model.fc3.register_full_backward_hook(get_grad)

criterion = nn.CrossEntropyLoss()

# Forward pass
output = model(input_tensor)
# print("Output shape:", output.shape)
# print("Output:", output)

# Compute the loss
loss = criterion(output, label_tensor)
print("Loss:", loss.item())

# Backward pass
model.zero_grad()  # Zero the gradient buffers
start = time.time()
loss.backward()    # Compute the gradients
end = time.time()
# print("total time = ", end - start)

# Print gradients for each layer
print("Gradients for each layer:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"Layer: {name}, Gradients: {param.grad.shape}")
        # if name == "fc3.weight":
        #     print(param.grad.detach().numpy())

# # Print gradients for the second convolutional layer's output
# print("Shape of dl/dx for the output of layer 1:", gradients.shape)
# print("Sum:", np.sum(gradients.detach().numpy()))
# print(gradients.detach().numpy())

# Manual gradient descent update
learning_rate = 0.01
with torch.no_grad():
    for param in model.parameters():
        param -= learning_rate * param.grad

# Print updated parameters
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
