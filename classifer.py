import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse

np.set_printoptions(threshold=np.inf)
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
    return np.pad(images[index], pad_width=((padding_size, padding_size),
                                            (padding_size, padding_size),
                                            (0,0)), mode="constant", constant_values=0)

def convolve(images, index, kernal, num_channels):
    output_image = np.zeros((32,32,num_channels))
    padded_image = zero_padding(images, index, 1)
    kernel_height, kernel_width = kernal.shape

    for i in range(32):
        for j in range(32):
            for c in range(num_channels):
                acc = 0 

                for m in range(kernel_height):
                    for  n in range(kernel_width): 
                       padded_i = i + m
                       padded_j = j + n

                       acc += kernal[m,n] * padded_image[padded_i, padded_j, c]

                       output_image[i,j,c] = np.clip(acc,0,255)
    return output_image


def max_pooling(output_image, pool_size=2,stride=2):
    height, width, num_channels = output_image.shape
    output_height = height // pool_size
    output_width = width // pool_size

    pooled_image = np.zeros((output_height,output_width,num_channels))
    for c in range(num_channels):
        for i in range(0, height - pool_size + 1, stride):
            for j in range(0, width - pool_size + 1, stride):
                pooled_image[i // stride, j // stride, c] = np.max(output_image[i:i + pool_size, j:j + pool_size, c])
    return pooled_image

def main(file_path, image_index):
    images, labels = load_batches(file_path)

    convoled_image = convolve(images, index=image_index, kernal=kernel_matrix, num_channels=3)
    print(convoled_image.shape)
    pooled_image = max_pooling(output_image=convoled_image)
    plt.imshow(pooled_image.astype(np.uint8))
    plt.title(f"image index:{image_index}")
    plt.show()   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="apply sharpening")
    parser.add_argument("index", type=int, help="Index of the cifar image")
    args = parser.parse_args()

    file_path = './cifar-10-batches-py/data_batch_1'

    main(file_path, args.index)