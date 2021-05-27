import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

X = np.array([[1,-1], [1,-1]])
true_val = np.array([[0.9], [0.05]])

#test example
# X = np.array([[1,2], [1,2]])
# true_val = np.array([[0.7], [0.7]])

examples = X.shape[0]

def initialize_params():

    # w01 shape (hidden layer, input layer)
    w01_ = np.array ([[0.3, 0.3],[0.3, 0.3]])
    # w12 shape (output layer, hidden layer)
    w12_ = np.array([[0.8, 0.8]])
    # bias shape (hidden layer, 1)
    b01_ = np.array([[0],[0]], dtype='float')
    # bias shape (output layer, 1)
    b12_ = np.array([[0]], dtype='float')

    return w01_, w12_, b01_, b12_

def activity(x,w,bias):
    assert x.shape == (2,1)
    return np.dot(w,x) + bias

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derv(x):
    return x * (1 - x)

def model(x, yhat, w01, w12, b01, b12, alpha = 1.0):
    
    Z1 = activity(x, w01, b01)
    #print(f"Z1 acvitity:\t{Z1}")

    output1 = sigmoid(Z1)
    #print(f"output1:\t{output1}")

    Z2 = activity(output1, w12, b12)
    #print(f"Z2 activity:\t{Z2:.5f}")

    output2 = sigmoid(Z2)
    # print("output2:\t", np.array2string(output2))

    pred_val = output2
    #print(f"sigmoid:\t{pred_val:.8f}")

    # calc error
    err = yhat - pred_val
    Error = .5 * err**2

    # calc delta for output layer
    delta12 = np.array((err * sigmoid_derv(pred_val)), dtype=float).reshape(1,1)

    # calc delta for hidden layer
    delta01 = sigmoid_derv(output1) * delta12 * w12.T

    # update weights from hidden to output layer
    w12 += alpha * delta12 * output1.T

    # update weights from input to hidden layer
    w01 += alpha * delta01 * x.T

    # update bias
    b12 += alpha * delta12
    b01 += alpha * delta01

    return pred_val, Error

def predict(x, yhat, cache):
    
    w01 = cache['w01']
    w12 = cache['w12']
    b01 = cache['b01']
    b12 = cache['b12']

    Z1 = activity(x, w01, b01)
    #print(f"Z1 acvitity:\t{Z1}")

    output1 = sigmoid(Z1)
    #print(f"output1:\t{output1}")

    Z2 = activity(output1, w12, b12)
    #print(f"Z2 activity:\t{Z2:.5f}")

    output2 = sigmoid(Z2)
    # print("output2:\t", np.array2string(output2))

    pred_val = output2
    #print(f"sigmoid:\t{pred_val:.8f}")

    # calc error
    err = yhat - pred_val
    Error = .5 * err**2

    return pred_val, Error

def train_method1(X, true_val, w01_, w12_, b01_, b12_, epochs= 15, print=False):

    cache = {}
    allError = []

    for i in range(epochs):
        #print("epoch:\t",i)

        for j in range(examples):
            x = X[:,j].reshape(2,1)
            if j == 0:
                #y_0, Error_0 = model(x,w01_0, w12_0, b01_0, b12_0)
                y_0, Error_0 = model(x, true_val[j], w01_, w12_, b01_, b12_)
                
                allError.append(float(Error_0))
            

            else:
                #y_1, Error_1 = model(x,w01_1, w12_1, b01_1, b12_1)
                y_1, Error_1 = model(x, true_val[j], w01_, w12_, b01_, b12_)

                allError.append(float(Error_1))

    cache['w01'] = w01_
    cache['w12'] = w12_
    cache['b01'] = b01_
    cache['b12'] = b12_

    if(print == True):
        print_graph(allError)

    return cache

def train_method2(X, true_val, w01_, w12_, b01_, b12_, epochs= 15, print=False):

    cache = {}
    allError = []

    for j in range(examples):

        for i in range(epochs):
            #print("epoch:\t",i)
            x = X[:,j].reshape(2,1)
            if j == 0:
                y_0, Error_0 = model(x, true_val[j], w01_, w12_, b01_, b12_)
                allError.append(float(Error_0))

            else:
                y_1, Error_1 = model(x, true_val[j], w01_, w12_, b01_, b12_)
                allError.append(float(Error_1))

    
    cache['w01'] = w01_
    cache['w12'] = w12_
    cache['b01'] = b01_
    cache['b12'] = b12_

    if(print == True):
        print_graph(allError)

    return cache


def print_graph(error):
    plt.plot(error)
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.show()

(w01_, w12_, b01_, b12_) = initialize_params()
cache = train_method1(X, true_val, w01_, w12_, b01_, b12_, print=True)
x = X[:,0].reshape(2,1)
pred, error = predict(x, true_val[0],  cache)
print("method1 input1")
print(pred, error)


x = X[:,1].reshape(2,1)
pred, error = predict(x, true_val[1], cache)
print("method1 input2")
print(pred, error)


cache = None
(w01_, w12_, b01_, b12_) = initialize_params()

cache = train_method2(X, true_val, w01_, w12_, b01_, b12_, print=True)
x = X[:,0].reshape(2,1)
pred, error = predict(x, true_val[0],  cache)
print("method2 input1")
print(pred, error)

x = X[:,1].reshape(2,1)
pred, error = predict(x, true_val[1], cache)
print("method2 input2")
print(pred, error)

