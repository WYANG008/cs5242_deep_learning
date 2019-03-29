import network
import utils
import numpy as np
import os


x = np.array([-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1])
x.shape = (14,1)
y = utils.convert_to_hot_vector(3,4)
y.shape = (4,1)

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

userhome = os.path.expanduser('~')
path_b_a = 'b-100-40-4.csv'
path_w_a = 'w-100-40-4.csv'
path_b_b = 'b-28-6-4.csv'
path_w_b = 'w-28-6-4.csv'
path_b_c = 'b-14-28-4.csv'
path_w_c = 'w-14-28-4.csv'

def output_gradient(x, y, sizes, shared_path, path_b, path_w):
    biases = [np.zeros((y,1)) for y in sizes[1:]]
    weights = [np.zeros((y,x)) for x, y in zip(sizes[:-1],sizes[1:])]
    with open(os.path.join(shared_path,path_b), newline='') as file:
        reader = csv.reader(file)
        for row, i in zip(reader,range(len(biases))):
            biases[i] = np.array([np.float32(v) for v in row[1:]])
            biases[i].shape = (len(row[1:]),1)
    with open(os.path.join(shared_path,path_w), newline='') as file:
        reader = csv.reader(file)
        indexes = [sum(sizes[0:i+1])  for i in range(len(sizes))]
        index = 0
        wts = [w.transpose() for w in weights]
        for row, i in zip(reader,range(sum(sizes[:-1]))):
            index = index + 1 if i >= indexes[index] else index
            index_w = i - indexes[index-1] if index > 0 else i
            wts[index][index_w] = [np.float32(v) for v in row[1:]]
    weights = [wt.transpose() for wt in wts]      
    network_a = network.Network(sizes, biases, weights)
    gradients_b, gradients_w = network_a.backpropagate(x, y)
    path_db = os.path.join(shared_path,'d'+path_b)
    path_dw = os.path.join(shared_path,'d'+path_w)
    with open(path_db, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for gb in gradients_b:
            n = [db[0] for db in gb]
            writer.writerow(n)
    with open(path_dw, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for gw in gradients_w:
            dw = gw.transpose()
            for row in dw:
                writer.writerow(row)

shared_path = userhome + r'/Downloads/assignment1/Question2_4/b'
output_gradient(x,y,get_size_a(),shared_path,path_b_a, path_w_a)
print('complete a')
output_gradient(x,y,get_size_b(),shared_path,path_b_b, path_w_b)
print('complete b')
output_gradient(x,y,get_size_c(),shared_path,path_b_c, path_w_c)
print('complete c')

shared_path = userhome + r'/Downloads/assignment1/Question2_4/c'
output_gradient(x,y,get_size_a(),shared_path,path_b_a, path_w_a)
print('complete test a')
output_gradient(x,y,get_size_b(),shared_path,path_b_b, path_w_b)
print('complete test b')
output_gradient(x,y,get_size_c(),shared_path,path_b_c, path_w_c)
print('complete test c')