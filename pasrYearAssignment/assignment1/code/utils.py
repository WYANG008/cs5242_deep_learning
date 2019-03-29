import random
import numpy as np

# eg. a_list[-1] = convert_to_hot_vector(np.argmax(a_list[-1]), 4)
def convert_to_hot_vector(y, num_class):
    v = np.zeros(num_class)
    v[y] = 1
    v.shape = (num_class,1)
    return v

# eg. convert_to_hot_vectors(np.array([1, 0, 3]), 4)
def convert_to_hot_vectors(y_s, num_class):
    m = np.zeros((y_s.size, num_class))
    m[np.arange(y_s.size), y_s] = 1
    return m

def relu(x):
    r = x * (x > 0)
    r.shape = x.shape
    return r

def relu_derivative(x):
    d = np.array([1 if i>0 else 0 for i in x])
    d.shape = x.shape
    return d 

def softmax(x):
    """Compute softmax values for each value in x. Minus max to avoid large values in intermidiate steps"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def cross_entropy(o,y):
    return -np.sum(np.multiply(y, np.log(o)))

# for debugging
def check_shape(list_of_arrays):
    shapes = [b.shape for b in list_of_arrays]
    print(shapes)
