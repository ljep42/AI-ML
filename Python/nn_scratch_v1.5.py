# CONVOLUTIONAL NETWORKS
import numpy as np
import sys
from keras.datasets import mnist

def tanh(x):
    return np.tanh(x)

def tanh2deriv(output):
    return 1 - (output ** 2)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# define TRAIN
images, labels = (x_train[0:1000].reshape(1000, 28 * 28) / 255,
                  y_train[0:1000])

# turn numbers into 1/0 output, hot encode
one_hot_labels = np.zeros((len(labels), 10))
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

# define TEST
test_images = x_test.reshape(len(x_test), 28 * 28) / 255
test_labels = np.zeros((len(y_test), 10))

for i, l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)
alpha, iterations = (2, 300)
pixels_per_image, num_labels = (784, 10)
batch_size = 128

input_rows = 28
input_cols = 28

kernel_rows = 3
kernel_cols = 3
num_kernels = 16

hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels

kernels = 0.02 * np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01
# weights_0_1 = 0.02*np.random.random((pixels_per_image,hidden_size))-0.01
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

#print(kernels.shape)
#print(weights_1_2.shape)


def get_image_section(layer, row_from, row_to, col_from, col_to):
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_from, col_to - col_from)


for j in range(iterations):
    correct_cnt = 0
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))

        # grab 128 rows of 784 columns in a batch
        layer_0 = images[batch_start:batch_end]

        # reshape the long string of 784 pixels back into a 28 * 28 matrix
        # batch still has 128 rows of 28*28 matrix
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        layer_0.shape

        # create list
        sects = list()
        # convolve the input layer and slice up in batches, add sections to list
        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0,
                                         row_start,
                                         row_start + kernel_rows,
                                         col_start,
                                         col_start + kernel_cols)
                sects.append(sect)
        # house keeping on sections
        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)
        # get the dot product between the kernel and the sections
        kernel_output = flattened_input.dot(kernels)
        # use the activation function but need to reshape, use -1 to let python figure it out
        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = softmax(np.dot(layer_1, weights_1_2))

        for k in range(batch_size):
            labelset = labels[batch_start + k:batch_start + k + 1]
            _inc = int(np.argmax(layer_2[k:k + 1]) ==
                       np.argmax(labelset))
            correct_cnt += _inc

        layer_2_delta = (labels[batch_start:batch_end] - layer_2) \
                        / (batch_size * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * \
                        tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        l1d_reshape = layer_1_delta.reshape(kernel_output.shape)
        k_update = flattened_input.T.dot(l1d_reshape)
        kernels -= alpha * k_update

    test_correct_cnt = 0

    for i in range(len(test_images)):

        layer_0 = test_images[i:i + 1]
        # layer_1 = tanh(np.dot(layer_0,weights_0_1))
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        layer_0.shape

        sects = list()
        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0,
                                         row_start,
                                         row_start + kernel_rows,
                                         col_start,
                                         col_start + kernel_cols)
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)

        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        layer_2 = np.dot(layer_1, weights_1_2)

        test_correct_cnt += int(np.argmax(layer_2) ==
                                np.argmax(test_labels[i:i + 1]))
    if (j % 10 == 0):
        sys.stdout.write("\n" + \
                         "I:" + str(j) + \
                         " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
                         " Train-Acc:" + str(correct_cnt / float(len(images))))