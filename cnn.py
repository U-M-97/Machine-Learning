#output size formula = (((w - k) + 2p) / s) + 1
import numpy as np

def conv(input_matrix, kernel, stride, padding):
    input_height, input_width = input_matrix.shape
    output_size = (((input_width - kernel.shape[1] + (2 * padding))) // stride) + 1
    input_matrix = np.pad(input_matrix, (padding, padding))
    out = np.zeros((output_size, output_size), dtype=float)
    for i in range(output_size):
        for j in range(output_size):
            sub_matrix = input_matrix[i*stride:i*stride + kernel.shape[1], j*stride:j*stride + kernel.shape[1]]
            out[i][j] = np.sum(sub_matrix * kernel)
    
    return out

def relu(x):
    return np.maximum(0, x)

def pooling(input, pool_size, stride):
    input_height, input_width = input.shape
    output_size = (((input_width - pool_size + (2 * 0))) // stride) + 1
    out = np.zeros((output_size, output_size), dtype=float)
    indices = []

    if output_size == 0:
        out = np.zeros((1, 1), dtype=float)
        out[0][0] = np.max(input)
        return out
    
    for i in range(output_size):
        for j in range(output_size):
            sub_matrix = input[i*stride:i*stride + pool_size, j*stride:j*stride + pool_size]
            out[i][j] = np.max(sub_matrix)
            indices.append(np.argmax(sub_matrix))

    return out, indices

def fully_connected(x, w, b):
    z = np.dot(x, w.T) + b
    return z

def softmax(x):
    # Subtracting the maximum value for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

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

def unpooling(dl_dz, prev_shape, pool_stride, pool_size, pooled_out_indices):
    dl_dc = np.zeros(prev_shape, dtype=float)
    indices_i = 0
    for a in range(dl_dz.shape[0]):
        for i in range(dl_dz.shape[1]):
            for j in range(dl_dz.shape[2]):
                sub_matrix = dl_dc[a, i*pool_stride:i*pool_stride + pool_size, j*pool_stride:j*pool_stride + pool_size]
                sub_matrix_shape = sub_matrix.shape
                sub_matrix = sub_matrix.flatten()
                sub_matrix[pooled_out_indices[a][indices_i]] = dl_dz[a, i, j]
                sub_matrix = sub_matrix.reshape(sub_matrix_shape)
                dl_dc[a, i*pool_stride:i*pool_stride + pool_size, j*pool_stride:j*pool_stride + pool_size] = sub_matrix
                indices_i += 1

        indices_i = 0
    return dl_dc

def conv_activation_gradient(feature_maps_layer):
    dc_dz = np.zeros(feature_maps_layer.shape, dtype=int)
    for i in range(feature_maps_layer.shape[0]):
        for j in range(feature_maps_layer.shape[1]):
            for k in range(feature_maps_layer.shape[2]):
                if feature_maps_layer[i, j, k] > 0:
                    dc_dz[i, j, k] = 1
    return dc_dz   

def kernels_biases_gradient(dl_dz, P):
    dl_dk = []
    dl_db_conv = []
    for i in range(dl_dz.shape[0]):
        channel_kernels = []
        channel_biases = []
        for j in range(P.shape[0]):
            channel_kernels.append(conv(P[j], dl_dz[i], stride=1, padding=0))

        dl_dk.append(channel_kernels)
        dl_db_conv.append(np.sum(dl_dz[i]))
    return dl_dk, dl_db_conv

def backward_propagation(act_output, y, act_fully_connected_layer2_out, w3, fully_connected_layer2_out, act_fully_connected_layer1_out, w2, fully_connected_layer1_out, flattened_out, w1, b1, pooled_outs2_shape, feature_activation_maps_layer_2_shape, pooled_outs2_indices, pool_size_2, pool_stride_2, feature_maps_layer_2, pooled_outs, kernels_layer_2, feature_activation_maps_layer_1_shape, pool_size_1, pool_stride_1, pooled_outs_indices, feature_maps_layer_1, x):
    dl_dz3 = act_output - y
    dl_dw3 = np.dot(dl_dz3.T, act_fully_connected_layer2_out)
    dl_db3 = dl_dz3.reshape(-1)

    dl_da2 = np.dot(dl_dz3, w3)
    dl_dz2 = dl_da2 * (fully_connected_layer2_out > 0)
    dl_dw2 = np.dot(dl_dz2.T, act_fully_connected_layer1_out)
    dl_db2 = dl_dz2.reshape(-1)

    dl_da1 = np.dot(dl_dz2, w2)
    dl_dz1 = dl_da1 * (fully_connected_layer1_out > 0)
    dl_dw1 = np.dot(dl_dz1.T, flattened_out)
    dl_db1 = dl_dz1.reshape(-1)

    dl_f = np.dot(dl_dz1, w1)
    dl_f_reshape = dl_f.reshape(pooled_outs2_shape)

    # unpooling layer 2
    dl_dc2 = unpooling(dl_f_reshape, feature_activation_maps_layer_2_shape, pool_stride_2, pool_size_2, pooled_outs2_indices)
    # gradient of layer 2 relu activation
    dc_dz2 = conv_activation_gradient(feature_maps_layer_2)
    dl_dz2_conv = dl_dc2 * dc_dz2

    dl_dk2, dl_db2_conv = kernels_biases_gradient(dl_dz2_conv, pooled_outs)
    
    dl_dx_conv_layer_2 = np.zeros((pooled_outs.shape), dtype=float)
    for j in range(kernels_layer_2.shape[1]):
        for i in range(dl_dz2_conv.shape[0]):
            kernel = np.rot90(kernels_layer_2[i, j])
            dl_dx_conv_layer_2[j] += conv(dl_dz2_conv[i], kernel, stride=1, padding=2)

    # unpooling layer 1
    dl_dc1 = unpooling(dl_dx_conv_layer_2, feature_activation_maps_layer_1_shape, pool_size_1, pool_stride_1, pooled_outs_indices)
    # gradient of layer 1 relu activation
    dc_dz1 = conv_activation_gradient(feature_maps_layer_1)
    dl_dz1_conv = dl_dc1 * dc_dz1

    dl_dk1, dl_db1_conv = kernels_biases_gradient(dl_dz1_conv, x.reshape(1, x.shape[0], x.shape[1]))

    return np.array(dl_dk1), np.array(dl_db1_conv), np.array(dl_dk2), np.array(dl_db2_conv), dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3

def forward_propagation(input_matrix, conv1_params, kernels_layer_1, biases_layer_1, conv1_stride, conv1_padding, pool1_size, pool1_stride, conv2_params, kernels_layer_2, biases_layer_2, conv2_stride, conv2_padding, pool2_size, pool2_stride, w1, b1, w2, b2, w3, b3):
    # Convolution Layer 1
    feature_maps_layer_1 = []
    feature_activation_maps_layer1 = []
    for i in range(conv1_params):
        conv_out = conv(input_matrix, kernels_layer_1[i][0], conv1_stride, conv1_padding)
        conv_out += biases_layer_1[i]
        feature_maps_layer_1.append(conv_out)
        act_out = relu(conv_out)
        feature_activation_maps_layer1.append(act_out)

    pooled_outs = []
    pooled_outs_indices = []
    for feature in feature_activation_maps_layer1:
        pooled_out, indices = pooling(feature, pool1_size, pool1_stride)
        pooled_outs.append(pooled_out)
        pooled_outs_indices.append(indices)

    # Convolutions Layer 2
    feature_maps_layer_2 = []
    feature_activation_maps_layer_2 = []
    for i in range(conv2_params):
        out_size = len(pooled_outs[1]) - kernels_layer_2.shape[2] + 1
        summed_conv = np.zeros((out_size, out_size), dtype=float)
        for j in range(len(pooled_outs)):
            summed_conv += conv(pooled_outs[j], kernels_layer_2[i][j], conv2_stride, conv2_padding)

        summed_conv += biases_layer_2[i]
        feature_maps_layer_2.append(summed_conv)
        act_out = relu(summed_conv)
        feature_activation_maps_layer_2.append(act_out)

    pooled_outs2 = []
    pooled_outs2_indices = []
    for feature in feature_activation_maps_layer_2:
        pooled_out, indices = pooling(feature, pool2_size, pool2_stride)
        pooled_outs2.append(pooled_out)
        pooled_outs2_indices.append(indices)

    pooled_outs2 = np.array(pooled_outs2)
    flattened_out = pooled_outs2.flatten().reshape(1, -1)
    input_size = flattened_out.shape[1]

    # fully connected layer 1
    fully_connected_layer1_out = fully_connected(flattened_out, w1, b1)
    act_fully_connected_layer1_out = relu(fully_connected_layer1_out)

    # fully connected layer 1
    fully_connected_layer2_out = fully_connected(act_fully_connected_layer1_out, w2, b2)
    act_fully_connected_layer2_out = relu(fully_connected_layer2_out)

    # output layer
    output = fully_connected(act_fully_connected_layer2_out, w3, b3)
    act_output = softmax(output)

    return act_output, act_fully_connected_layer2_out, fully_connected_layer2_out, act_fully_connected_layer1_out, fully_connected_layer1_out, flattened_out, pooled_outs2, feature_activation_maps_layer_2, pooled_outs2_indices, feature_maps_layer_2, pooled_outs, feature_activation_maps_layer1, pooled_outs_indices, feature_maps_layer_1

def crossEntropyLoss(output, y):
    m = y.shape[0]
    log_probs = -np.log(output[np.arange(m), y.argmax(axis=1)])
    loss_value = np.sum(log_probs) / m
    return loss_value

def update_parameters(learning_rate, w3, dl_dw3, b3, dl_db3, w2, dl_dw2, b2, dl_db2, w1, dl_dw1, b1, dl_db1, kernels_layer_2, dl_dk2, biases_layer_2, dl_db2_conv, kernels_layer_1, dl_dk1, biases_layer_1, dl_db1_conv):
    w3 -= (learning_rate * dl_dw3)
    b3 -= (learning_rate * dl_db3)
    w2 -= (learning_rate * dl_dw2)
    b2 -= (learning_rate * dl_db2)
    w1 -= (learning_rate * dl_dw1)
    b1 -= (learning_rate * dl_db1)
    kernels_layer_2 -= (learning_rate * dl_dk2)
    biases_layer_2 -= (learning_rate * dl_db2_conv)
    kernels_layer_1 -= (learning_rate * dl_dk1)
    biases_layer_1 -= (learning_rate * dl_db1_conv)

    return w3, b3, w2, b2, w1, b1, kernels_layer_2, biases_layer_2, kernels_layer_1, biases_layer_1

def train(input_matrix, y, learning_rate, iterations, conv1_params, conv1_stride, conv1_padding, pool1_size, pool1_stride, pool1_padding, conv2_params, conv2_stride, conv2_padding, pool2_size, pool2_stride, pool2_padding, kernel_size, flatten_layer_size, fc1_neurons, fc2_neurons, output_layer):
    kernels_layer_1, biases_layer_1 = initialize_kernels(conv1_params, 1, kernel_size)
    kernels_layer_2, biases_layer_2 = initialize_kernels(conv2_params, conv1_params, kernel_size)
    w1, b1 = initialize_params(flatten_layer_size, fc1_neurons)
    w2, b2 = initialize_params(fc1_neurons, fc2_neurons)
    w3, b3 = initialize_params(fc2_neurons, output_layer)

    for itr in range(iterations):
        act_output, act_fully_connected_layer2_out, fully_connected_layer2_out, act_fully_connected_layer1_out, fully_connected_layer1_out, flattened_out, pooled_outs2, feature_activation_maps_layer_2, pooled_outs2_indices, feature_maps_layer_2, pooled_outs, feature_activation_maps_layer1, pooled_outs_indices, feature_maps_layer_1 = forward_propagation(input_matrix, conv1_params, kernels_layer_1, biases_layer_1, conv1_stride, conv1_padding, pool1_size, pool1_stride, conv2_params, kernels_layer_2, biases_layer_2, conv2_stride, conv2_padding, pool2_size, pool2_stride, w1, b1, w2, b2, w3, b3)
        loss_value = crossEntropyLoss(act_output, y)
        dl_dk1, dl_db1_conv, dl_dk2, dl_db2_conv, dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3 = backward_propagation(act_output, y, act_fully_connected_layer2_out, w3, fully_connected_layer2_out, act_fully_connected_layer1_out, w2, fully_connected_layer1_out, flattened_out, w1, b1, pooled_outs2.shape, np.array(feature_activation_maps_layer_2).shape, np.array(pooled_outs2_indices), pool2_size, pool2_stride, np.array(feature_maps_layer_2), np.array(pooled_outs), np.array(kernels_layer_2), np.array(feature_activation_maps_layer1).shape, pool1_size, pool1_stride, pooled_outs_indices, np.array(feature_maps_layer_1), input_matrix)
        w3, b3, w2, b2, w1, b1, kernels_layer_2, biases_layer_2, kernels_layer_1, biases_layer_1 = update_parameters(learning_rate, w3, dl_dw3, b3, dl_db3, w2, dl_dw2, b2, dl_db2, w1, dl_dw1, b1, dl_db1, kernels_layer_2, dl_dk2, biases_layer_2, dl_db2_conv, kernels_layer_1, dl_dk1, biases_layer_1, dl_db1_conv)

        if itr % 100 == 0:
            print(f"Iteration {itr}, Loss: {loss_value}")
            print(np.sum(w3))

    return kernels_layer_1, biases_layer_1, kernels_layer_2, biases_layer_2, w1, b1, w2, b2, w3, b3

def main():
    np.random.seed(0)

    # input_matrix = np.random.normal(loc=0.5, scale=0.25, size=(28, 28))
    input_matrix = np.random.normal(loc=0.5, scale=0.25, size=(28, 28))
    # Normalize the matrix
    normalized_matrix = (input_matrix - np.min(input_matrix)) / (np.max(input_matrix) - np.min(input_matrix))
    print(np.sum(normalized_matrix))

    num_classes = 10
    y = np.zeros(num_classes, dtype=int)
    y[5] = 1
    y = y.reshape(1, -1)
    
    learning_rate = 0.01
    iterations = 20000
    conv1_params = 32
    conv1_stride = 1
    conv1_padding = 0
    pool1_size = 2
    pool1_stride = 2
    pool1_padding = 0
    conv2_params = 64
    conv2_stride = 1
    conv2_padding = 0
    pool2_size = 2
    pool2_stride = 2
    pool2_padding = 0
    kernel_size = 3
    fc1_neurons = 128
    fc2_neurons = 100
    output_layer = num_classes

    conv1_output_size = ((input_matrix.shape[0] - kernel_size + (2 * conv1_padding)) // conv1_stride) + 1
    conv1_pooled_size = ((conv1_output_size - pool1_size + (2 * pool1_padding)) // pool1_stride) + 1
    conv2_output_size = ((conv1_pooled_size - kernel_size + (2 * conv1_padding)) // conv1_stride) + 1
    conv2_pooled_size = ((conv2_output_size - pool2_size + (2 * pool1_padding)) // pool1_stride) + 1
    flatten_layer_size = conv2_params * conv2_pooled_size * conv2_pooled_size

    kernels_layer_1, biases_layer_1, kernels_layer_2, biases_layer_2, w1, b1, w2, b2, w3, b3 = train(normalized_matrix, y, learning_rate, iterations, conv1_params, conv1_stride, conv1_padding, pool1_size, pool1_stride, pool1_padding, conv2_params, conv2_stride, conv2_padding, pool2_size, pool2_stride, pool2_padding, kernel_size, flatten_layer_size, fc1_neurons, fc2_neurons, output_layer)

main()