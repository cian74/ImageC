import pickle
import matplotlib.pyplot as plt
import numpy as np

file_path = './cifar-10-batches-py/data_batch_1'
kernel_matrix = np.array([[0,-1,0],
                          [-1,5,-1],
                          [0,-1,0]]) 
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

def view_matrix(images,index):
    for row in images[index]:
        print(row)

def display(images, index):
    plt.imshow(images[index])
    plt.show()
 
def view_dest_matrix(images, index):
    for row in images[index]:
        print(row) 
    
def zero_padding(images, index, padding_size):
    return np.pad(images[index], pad_width=padding_size, mode="constant", constant_values=0)

def convolve(images, index, kernal, num_channels):
    output_image = np.zeros((32,32,num_channels))
    padded_image = zero_padding(images, index, 1)

    print(padded_image)

    """ for row in images[index]:
        for pixel in row:

            print(pixel)
            for rbg in num_channels:

                acc = 0

                for kernal_rows in kernal:
                    for element in kernal_rows: """
                        
images, labels = load_batches(file_path)
convolve(images,index=5,kernal=kernel_matrix, num_channels=3)
#view_matrix(images, index=4)
#for row in kernel_matrix:
#    print(row)
#display(images, index=5)