# HW#5 Machine Learning 10-601, Meiirbek Islamov
# Neural Networks

# import the necessary libraries
import sys
import numpy as np
import csv

args = sys.argv
assert(len(args) == 10)
train_input = args[1] # Path to the training data .csv file
valid_input = args[2] # Path to the validation input .csv file
train_out = args[3] # Path to output .labels file to which the prediction on the training data should be written
valid_out = args[4] # Path to output .labels file to which the prediction on the validation data should be written
metrics_out = args[5] # Path of the output .txt file to which metrics such as train and validation error should be written
num_epoch = int(args[6]) # Integer specifying the number of times backpropogation loops through all of the training data
hidden_units = int(args[7]) # Positive integer specifying the number of hidden units
init_flag = int(args[8]) # Integer taking value 1 or 2 that specifies whether to use RANDOM or ZERO initial- ization
learning_rate = float(args[9]) # Float value specifying the learning rate for SGD

# Functions

# Read input csv file
def read(input):
    with open(input, newline='') as f_in:
        read_csv = csv.reader(f_in)
        data = np.array(list(read_csv))
    data_int = data.astype(int)
    return data_int[:, 1:], data_int[:, 0]

def insert_zeros(array):
    array = np.insert(array, 0, np.zeros([len(array)]), axis=1)
    return array

def insert_ones(array):
    array = np.insert(array, 0, np.ones([len(array)]), axis=1)
    return array

# Forward computation
def LinearForward(x, alpha):
    return np.dot(x,alpha.T)

def SigmoidForward(a):
    return 1/(1 + np.exp(-a))

def SoftmaxForward(b):
    return np.exp(b)/np.sum(np.exp(b))

def CrossEntropyForward(y, y_hat):
    return np.log(y_hat[0, y])

def one_hot(y):
    array = np.zeros((1, 10))
    array[0][y] = 1
    return array

# Backpropagations
def CrossEntropyBackward(y, y_hat):
    return -y/y_hat

def SoftmaxBackward(b, y_hat, g_y_hat):
    array = np.zeros((10, 10))
    for i in range(len(y_hat[0])):
        for j in range(len(y_hat[0])):
            array[i][j] = (y_hat[0][i] * (- y_hat[0][j]))
    for i in range(len(y_hat[0])):
        array[i][i] = y_hat[0][i] * (1 - y_hat[0][i])
    g_b = np.dot(g_y_hat, array)
    return g_b

def LinearBackward_hidden(z, beta, g_b):
    g_z = np.dot(g_b, beta)
    beta = np.tile(z, (10, 1))
    g_beta = g_b.T * beta
    return g_beta, g_z

def SigmoidBackward(a, z, g_z):
    return g_z*z*(1-z)

def LinearBackward(a, alpha, g_a, D):
    g_x = np.dot(g_a, alpha)
    alpha = np.tile(a, (D, 1))
    g_alpha = g_a.T * alpha
    return g_alpha, g_x

# Stochastic Gradient Descent
# Single SGD step
def sgd_single(alpha, beta, example, label, learning_rate, D):
    # Forward computation
    a = LinearForward(example, alpha)
    z = SigmoidForward(a)
    z = insert_ones(z)
    b = LinearForward(z, beta)
    y_hat = SoftmaxForward(b)
    y = one_hot(label)
    J = CrossEntropyForward(label,y_hat)


    # Backpropagation
    g_y_hat = CrossEntropyBackward(y, y_hat)
    g_b = SoftmaxBackward(b, y_hat, g_y_hat)
    g_beta, g_z = LinearBackward_hidden(z, beta, g_b)
    g_a = SigmoidBackward(a, z, g_z)
    g_alpha, g_x = LinearBackward(example, alpha, g_a[:, 1:], D)

    alpha = alpha - learning_rate * g_alpha
    beta = beta - learning_rate * g_beta

    return alpha, beta

# Many SGD steps
def sgd_many(alpha, beta, train_data, valid_data, label_train, label_valid, learning_rate, num_epoch, D):
    objective_function_train = []
    objective_function_valid = []
    for i in range(num_epoch):
        for j, item in enumerate(label_train):
            alpha, beta = sgd_single(alpha, beta, np.atleast_2d(train_data[j]), label_train[j], learning_rate, D)
        objective_function_train.append(Objective(alpha, beta, train_data, label_train))
        objective_function_valid.append(Objective(alpha, beta, valid_data, label_valid))
    return alpha, beta, objective_function_train, objective_function_valid

# Objective Function
def Objective(alpha, beta, train_data, label):
    product = 0
    for i, item in enumerate(label):
        a = LinearForward(np.atleast_2d(train_data[i]), alpha)
        z = SigmoidForward(a)
        z = insert_ones(z)
        b = LinearForward(z, beta)
        y_hat = SoftmaxForward(b)
        y = one_hot(label[i])
        J = CrossEntropyForward(label[i], y_hat)
        product += J
    return -(1/len(label)) * product

# Predict labels
def predict_labels(alpha, beta, train_data):
    labels = []
    for i, item in enumerate(train_data):
        a = LinearForward(np.atleast_2d(train_data[i]), alpha)
        z = SigmoidForward(a)
        z = insert_ones(z)
        b = LinearForward(z, beta)
        y_hat = SoftmaxForward(b)
        result = np.where(y_hat[0] == np.amax(y_hat[0]))
        labels.append(result[0][0])
    return labels

def calculate_error(label_true, label_predicted):
    n = 0
    for i, item in enumerate(label_true):
        if item != label_predicted[i]:
            n += 1
    error = n/len(label_true)
    return error

def write_labels(predicted_label, filename):
    with open(filename, 'w') as f_out:
        for label in predicted_label:
            f_out.write(str(label) + '\n')

def write_error(objective_function_train, objective_function_valid, train_error, valid_error, filename):
    with open(filename, 'w') as f_out:
        for i, item in enumerate(objective_function_train):
            f_out.write("epoch=" + str(i + 1) + " " + "crossentropy(train): " + str(item) + "\n")
            f_out.write("epoch=" + str(i + 1) + " " + "crossentropy(validation): " + str(objective_function_valid[i]) + "\n")
        f_out.write("error(train): " + str(train_error) + "\n")
        f_out.write("error(validation): " + str(valid_error) + "\n")

# RANDOM or ZERO Initialization
def initialization(init_flag, D, M, K):
    if init_flag == 2:
        alpha = np.zeros((D, M))
        beta = np.zeros((K, D))
    else:
        alpha = np.random.uniform(-0.1,0.1,((D, M)))
        beta = np.random.uniform(-0.1,0.1,((K, D)))
    alpha = insert_zeros(alpha)
    beta = insert_zeros(beta)

    return alpha, beta

data_train, label_train = read(train_input)
data_valid, label_valid = read(valid_input)
# Initialization of the weight vectors, alpha and beta
M = len(data_train[0])
D = hidden_units
K = 10
alpha, beta = initialization(init_flag, D, M, K)
train_with_bias = insert_ones(data_train)
valid_with_bias = insert_ones(data_valid)
alpha, beta, obj_func_train, obj_func_valid = sgd_many(alpha, beta, train_with_bias, valid_with_bias, label_train, label_valid, learning_rate, num_epoch, D)
label_pred_train = predict_labels(alpha, beta, train_with_bias)
label_pred_valid = predict_labels(alpha, beta, valid_with_bias)
train_error = calculate_error(label_train, label_pred_train)
valid_error = calculate_error(label_valid, label_pred_valid)
write_labels(label_pred_train, train_out)
write_labels(label_pred_valid, valid_out)
write_error(obj_func_train, obj_func_valid, train_error, valid_error, metrics_out)
