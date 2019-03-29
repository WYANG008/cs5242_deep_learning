import random
import numpy as np
import utils


class Network(object):
    def __init__(self, sizes, initial_b=None, initial_w=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = initial_b if initial_b else [np.random.random((y,1)) - 0.5 for y in sizes[1:]]
        self.weights = initial_w if initial_w else [np.random.random((y,x)) - 0.5 for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, a):
        for b, w in zip(self.biases[0:-1], self.weights[0:-1]):
            a = utils.relu(np.dot(w,a) + b)
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        a = utils.softmax(z)
        return a
    
    def stochastic_gradient_descent(self, training_data, iterations, mini_batch_size, learning_rate, test_data=None):
        """training_data is a list of tupes (x,y). ''test_data'' is used for cost & accuracy evaluation"""
        n = len(training_data)
        training_cost = np.zeros(iterations)
        test_cost = np.zeros(iterations)
        training_accuracy = np.zeros(iterations)
        test_accuracy = np.zeros(iterations)
        for i in range(iterations):
            # evaluate cost & accuracy for training data and test data
            training_cost[i], training_accuracy[i] = self.evaluate(training_data)
            print("Iter", i,": TrC: ", training_cost[i], "; TrA: ",training_accuracy[i])
            if test_data:
                test_cost[i], test_accuracy[i] = self.evaluate(test_data)
                print("Iter ", i,": TeC: ", test_cost[i], "; TeA: ", test_accuracy[i])
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_weights_biases(mini_batch, learning_rate)
        return (training_cost, training_accuracy, test_cost, test_accuracy)
    
    def update_weights_biases(self, mini_batch, learning_rate):
        """The ``mini_batch`` is a list of tuples ``(x, y)``."""
        sum_gradients_b = [np.zeros(b.shape) for b in self.biases]
        sum_gradients_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            gradients_b, gradients_w = self.backpropagate(x, y)
            sum_gradients_b = [sum_gb + gb for sum_gb, gb in zip(sum_gradients_b, gradients_b)]
            sum_gradients_w = [sum_gw + gw for sum_gw, gw in zip(sum_gradients_w, gradients_w)]
        self.biases = [b - learning_rate/len(mini_batch)*gb for b, gb in zip(self.biases, sum_gradients_b)]
        self.weights = [w - learning_rate/len(mini_batch)*gw for w, gw in zip(self.weights, sum_gradients_w)]
    
    def backpropagate(self, x, y):
        gradients_b = [np.zeros(b.shape) for b in self.biases]
        gradients_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        a = x
        a_list = [x]
        z_list = []
        for b, w in zip(self.biases[0:-1], self.weights[0:-1]):
            z = np.dot(w, a) + b
            a = utils.relu(z)
            a_list.append(a)
            z_list.append(z)
        z = np.dot(self.weights[-1], a_list[-1]) + self.biases[-1]
        a = utils.softmax(z)
        a_list.append(a)
        z_list.append(z)
        # backward
        # for softmax-cross-entropy layer: delta in last layer = result - ground truth
        delta = a_list[-1] - y
        # update b and w for the last layer L
        gradients_b[-1] = delta
        gradients_w[-1] = np.dot(delta, a_list[-2].transpose())
        # update b and w for the rest of layers L-1, L-2, ... 
        for l in range(2, self.num_layers):
            z = z_list[-l]  # lth last layer of z
            r_derivative = utils.relu_derivative(z)
            # update delta based on delta(l) = transpose of w(l+1) * delta(l+1)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * r_derivative
            gradients_b[-l] = delta
            gradients_w[-l] = np.dot(delta, a_list[-l-1].transpose())
        return (gradients_b, gradients_w)
    
    def evaluate(self, data):
        output = [self.feedforward(x) for x, y in data]
        cost = [utils.cross_entropy(o, d[1]) for o, d in zip(output, data)]
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        accuracy = sum(int(x == y) for (x, y) in results)/len(results)
        avg_cost = sum(cost)/len(cost)
        return (avg_cost, accuracy)
