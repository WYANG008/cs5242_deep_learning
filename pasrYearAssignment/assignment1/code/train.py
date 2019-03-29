import random
import numpy as np
from numpy import genfromtxt
import os
import matplotlib.pyplot as plt
import network
import utils

def get_size_a():
    size_a = [14,100,40,4]
    return size_a
    
def get_size_b():
    size_b = [14]
    for i in range(6):
        size_b.append(28)
    size_b.append(4)
    return size_b

def get_size_c():
    size_c = [14]
    for i in range(28):
        size_c.append(14)
    size_c.append(4)
    return size_c

def save_to_file(title, training_cost, training_accuracy, test_cost, test_accuracy):
    userhome = os.path.expanduser('~')
    path = userhome + r'/Downloads/e0047338/toDelete'
    filename = os.path.join(path, title + '.txt')
    np.savetxt(filename, (training_cost, training_accuracy,test_cost,test_accuracy), delimiter=',', fmt='%1.4f')


def plot_graph(title, y_1, y_2, size, label_1, label_2, x_label, y_label, y_lim = None):
    plt.figure(figsize=(15,7))
    plt.title(title)
    plt.plot(np.arange(0, size, 1), y_1, 'o-', label = label_1)
    plt.plot(np.arange(0, size, 1), y_2, 'o-', label = label_2)
    plt.grid() 
    if y_lim:
        plt.ylim(y_lim)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.legend(loc='best', fontsize=16)

# prepare data
userhome = os.path.expanduser('~')
shared_path = userhome + r'/Downloads/assignment1/Question2_123'
path_training_x = 'x_train.csv'
path_training_y = 'y_train.csv'
path_test_x = 'x_test.csv'
path_test_y = 'y_test.csv'
training_X =  genfromtxt(os.path.join(shared_path, path_training_x),delimiter=",")
training_Y =  genfromtxt(os.path.join(shared_path, path_training_y),delimiter=",")
training_data = [(x.reshape(14,1), utils.convert_to_hot_vector(int(y),4)) for x, y in zip(training_X,training_Y)]
test_X =  genfromtxt(os.path.join(shared_path, path_test_x),delimiter=",")
test_Y =  genfromtxt(os.path.join(shared_path, path_test_y),delimiter=",")
test_data = [(x.reshape(14,1), utils.convert_to_hot_vector(int(y),4)) for x, y in zip(test_X, test_Y)]


# training and evaluation for network 14-100-40-4
network_a = network.Network(get_size_a())
training_cost, training_accuracy, test_cost, test_accuracy = network_a.stochastic_gradient_descent(training_data, 1000, 100, 0.1, test_data)
# save cost and accuracy to file
save_to_file('14-100-40-4', training_cost, training_accuracy, test_cost, test_accuracy)
# plot cost and accuracy graph
cost_title = 'Nerual Network 14-100-40-4 Cost Graph'
accuracy_title = 'Nerual Network 14-100-40-4 Accuracy Graph'
plot_graph(cost_title, training_cost[:500], test_cost[:500], 500, 'training_cost plot', 'test_cost plot', '$iteration$', '$cost$',(-1,10))
plot_graph(accuracy_title, training_accuracy[:500], test_accuracy[:500], 500, 'training_accuracy plot', 'test_accuracy plot', '$iteration$', '$accuracy$',(0,1.1))


# training and evaluation for network 14-28-6-4
network_b = network.Network(get_size_b())
training_cost, training_accuracy, test_cost, test_accuracy = network_b.stochastic_gradient_descent(training_data, 1000, 100, 0.1, test_data)
# save cost and accuracy to file
save_to_file('14-28-6-4', training_cost, training_accuracy, test_cost, test_accuracy)
# plot cost and accuracy graph
cost_title = 'Nerual Network 14-28-6-4 Cost Graph'
accuracy_title = 'Nerual Network 14-28-6-4 Accuracy Graph'
plot_graph(cost_title, training_cost[:500], test_cost[:500], 500, 'training_cost plot', 'test_cost plot', '$iteration$', '$cost$',(-1,10))
plot_graph(accuracy_title, training_accuracy[:500], test_accuracy[:500], 500, 'training_accuracy plot', 'test_accuracy plot', '$iteration$', '$accuracy$',(0,1.1))


# training and evaluation for network 14-28-6-4
network_c = network.Network(get_size_c())
training_cost, training_accuracy, test_cost, test_accuracy = network_c.stochastic_gradient_descent(training_data, 1000, 100, 0.1, test_data)
# save cost and accuracy to file
save_to_file('14-14-28-4', training_cost, training_accuracy, test_cost, test_accuracy)
# plot cost and accuracy graph
cost_title = 'Nerual Network 14-14-28-4 Cost Graph'
accuracy_title = 'Nerual Network 14-14-28-4 Accuracy Graph'
plot_graph(cost_title, training_cost[:500], test_cost[:500], 500, 'training_cost plot', 'test_cost plot', '$iteration$', '$cost$',(-1,10))
plot_graph(accuracy_title, training_accuracy[:500], test_accuracy[:500], 500, 'training_accuracy plot', 'test_accuracy plot', '$iteration$', '$accuracy$',(0,1.1))