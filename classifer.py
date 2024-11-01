import pickle
import matplotlib.pyplot as plt
import numpy as np

file_path = './cifar-10-batches-py/data_batch_1'
val = 0
traverse_matrix = [[val for _ in range(3)] for _ in range(3)]

def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        print("unpickling")
    return dict

def load_batches(file_path):
    batch = unpickle(file_path)
    data = batch[b'data']
    labels = batch[b'labels']
    images = data.reshape(-1,3,32,32).transpose(0,2,3,1)
    return images, labels

def display(images, index):
    plt.imshow(images[index])
    plt.show()

images, labels = load_batches(file_path)
display(images, index=5)