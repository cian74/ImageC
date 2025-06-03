import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse


np.set_printoptions(threshold=np.inf)
file_path = './cifar-10-batches-py/'
kernel_matrix = np.array([[0,-1,0],
                          [-1,5,-1],
                          [0,-1,0]]) 

#cifar10 only has images containing these classes
classes = [
    'plane', 'car', 'bird', 'cat','deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

dense_weights  = np.random.rand(8 * 8 * 3, 10) * 0.01
dense_biases = np.zeros(10)

def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        print("unpickling")
    return dict

def load_batches(file_path):
    batch = unpickle(file_path)
    data = batch[b'data']
    labels = np.array(batch[b'labels'])
    images = data.reshape(-1,3,32,32).transpose(0,2,3,1)
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    return images, labels

def load_all_batches(file_path):
    all_images = []
    all_labels = []
    for i in range(1, 6):
        batch_images, batch_labels = load_batches(f"{file_path}/data_batch_{i}")
        all_labels.append(batch_labels)
        all_images.append(batch_images)
    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
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

def relu(image):
    return np.maximum(0, image)

def flatten(image):
    return image.flatten()

def convolve(images, index, kernel, num_channels):
    image = zero_padding(images, index, 1)
    output_image = np.zeros((32, 32, num_channels))
    for c in range(num_channels):
        output_image[:, :, c] = convolve2d(image[:, :, c], kernel)
    return np.clip(output_image, 0, 255)


def convolve2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    # Output dimensions
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output


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

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def dense(input_vector, weights, bias):
    return np.dot(input_vector, weights) + bias

def forward(images, index, kernel, num_channels):
    conv1 = convolve(images, index, kernel, num_channels)
    pooled1 = max_pooling(conv1)
    relu1 = relu(pooled1)

    # Pad relu1 before second conv
    padded_relu1 = np.pad(relu1, ((1, 1), (1, 1), (0, 0)), mode='constant')
    second_conv = np.zeros_like(relu1)
    for c in range(num_channels):
        second_conv[:, :, c] = convolve2d(padded_relu1[:, :, c], kernel)
    relu2 = relu(second_conv)
    pooled2 = max_pooling(relu2)

    pooled2 /= 255.0
    flattened = flatten(pooled2)

    global dense_weights
    if dense_weights.shape != (flattened.shape[0], 10):
        dense_weights = np.random.randn(flattened.shape[0], 10) * 0.01

    logits = dense(flattened, dense_weights, dense_biases)
    probs = softmax(logits)
    return probs, flattened

def backward(flattened_image, output_probs, true_label, learning_rate=0.01):
    global dense_weights, dense_biases

    # Compute the gradient of the loss w.r.t. the output layer
    true_label_one_hot = np.zeros(10)
    true_label_one_hot[true_label] = 1
    gradient_output = output_probs - true_label_one_hot

    # Update weights and biases
    gradient_weights = np.outer(flattened_image, gradient_output)
    gradient_biases = gradient_output

    dense_weights -= learning_rate * gradient_weights
    dense_biases -= learning_rate * gradient_biases

def train(images, labels, kernal, num_channels, epochs=10, learning_rate=0.01, log_interval=100):
    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        images = images[indices]
        labels = labels[indices]
        
        for index, true_label in enumerate(labels):
            output_probs, flattened_image = forward(images, index, kernal, num_channels)
            
            loss = cross_entropy_loss(output_probs, true_label)
            total_loss += loss

            backward(flattened_image, output_probs, true_label, learning_rate)

            predicted_label = np.argmax(output_probs)
            if predicted_label == true_label:
                correct += 1

            if (index + 1) % log_interval == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Step {index + 1}/{len(labels)}, "
                    f"Loss: {total_loss / (index + 1):.4f}, Accuracy: {correct / (index + 1) * 100:.2f}%")

        accuracy = correct / len(labels)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")


def predict(images, index, kernal, num_channels):
    output_probs, _ = forward(images, index, kernal, num_channels)
    predicted_label = np.argmax(output_probs)
    return predicted_label

def evaluate(images, labels, kernal, num_channels):
    correct = 0
    for index, label in enumerate(labels):
        predicted_label = predict(images, index, kernal, num_channels)
        if predicted_label == label:
            correct += 1
    accuracy = correct / len(labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def cross_entropy_loss(output_probs, true_label):
    epsilon = 1e-12
    output_probs = np.clip(output_probs, epsilon, 1.0)
    return -np.log(output_probs[true_label])

def main(file_path, image_index):
    images, labels = load_all_batches('./cifar-10-batches-py')
    images = images / 255.0
    print("training.")
    train(images, labels, kernal=kernel_matrix, num_channels=3, epochs=50, learning_rate=0.01)

    print("evaluating.")
    evaluate(images / 255.0, labels, kernal=kernel_matrix, num_channels=3)

    predicted_label = predict(images, image_index, kernal=kernel_matrix, num_channels=3)
    
    print(f"Predicted Label: {list(classes)[predicted_label]}")
    print(f"True Label: {list(classes)[labels[image_index]]}")
    
    plt.imshow(images[image_index])
    plt.title(f"Predicted: {list(classes)[predicted_label]}, True: {list(classes)[labels[image_index]]}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="apply sharpening")
    parser.add_argument("index", type=int, help="Index of the cifar image")
    args = parser.parse_args()

    file_path = './cifar-10-batches-py/data_batch_1'

    main(file_path, args.index)
