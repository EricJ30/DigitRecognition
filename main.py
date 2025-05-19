from os.path import join
import struct
from array import array
import pandas as pd
import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import pickle
import tkinter as tk
from tkinter import Canvas, Button, Frame, Label, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm

# main.py

from models.mcp import MCP
from models.dense import Dense
from models.relu import ReLU
from models.layer import Layer

from utils.mnist_dataloader import *
from utils.math_utils import *

from gui.digit_drawing_app import DigitDrawingApp

#import DigitDrawingApp
'''
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)  

def softmax_crossentropy_with_logits(logits, reference_answers):
    # Compute crossentropy from logits[batch,n_classes] and ids of correct answers                 
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]    
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))    
    return xentropy

def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)    
    return (- ones_for_answers + softmax) / logits.shape[0]

class Layer(object):
    def __init__(self):        
        pass
    
    def forward(self, input):
        return input
    
    def backward(self, input, grad_output):
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)


class ReLU(Layer):
    def __init__(self):
        # ReLU layer simply applies elementwise rectified linear unit to all inputs
        pass
    
    def forward(self, input):
        # Apply elementwise ReLU to [batch, input_units] matrix
        relu_forward = np.maximum(0, input)
        return relu_forward
    
    def backward(self, input, grad_output):
        # Compute gradient of loss w.r.t. ReLU input
        relu_grad = input > 0
        return grad_output * relu_grad

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        # A dense layer is a layer which performs a learned affine transformation: f(x) = <W*x> + b
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2 / (input_units + output_units)), size=(input_units, output_units))
        self.biases = np.zeros(output_units)
    
    def forward(self, input):
        # Perform an affine transformation: f(x) = <W*x> + b        
        # input shape: [batch, input_units]
        # output shape: [batch, output units]        
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, input, grad_output):
        # compute d f / d x = d f / d dense * d dense / d x where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)
        
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        # Here we perform a stochastic gradient descent step. 
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input

    
class MCP(object):
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, X):
        # Compute activations of all network layers by applying them sequentially.
        # Return a list of activations for each layer. 
        activations = []
        input = X
        
        # Looping through each layer
        for l in self.layers:
            activations.append(l.forward(input))
            # Updating input to last layer output
            input = activations[-1]
    
        assert len(activations) == len(self.layers)
        return activations
    
    def get_layer_activations(self, X):
        """Get all layer activations for visualization"""
        layer_activations = []
        input_data = X
        
        # Add input layer
        layer_activations.append(input_data)
        
        # Get activations for all layers
        for layer in self.layers:
            input_data = layer.forward(input_data)
            layer_activations.append(input_data)
            
        return layer_activations
    
    def train_batch(self, X, y):
        # Train our network on a given batch of X and y.
        # We first need to run forward to get all layer activations.
        # Then we can run layer.backward going from last to first layer.
        # After we have called backward for all layers, all Dense layers have already made one gradient step.
        
        layer_activations = self.forward(X)
        layer_inputs = [X] + layer_activations  # layer_input[i] is an input for layer[i]
        logits = layer_activations[-1]
        
        # Compute the loss and the initial gradient    
        y_argmax = y.argmax(axis=1)        
        loss = softmax_crossentropy_with_logits(logits, y_argmax)
        loss_grad = grad_softmax_crossentropy_with_logits(logits, y_argmax)
    
        # Propagate gradients through the network
        # Reverse propogation as this is backprop
        for layer_index in range(len(self.layers))[::-1]:
            layer = self.layers[layer_index]        
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)  # grad w.r.t. input, also weight updates
        
        return np.mean(loss)
    
    def train(self, X_train, y_train, n_epochs=25, batch_size=32):
        train_log = []        
        
        for epoch in range(n_epochs):        
            for i in range(0, X_train.shape[0], batch_size):
                # Get pair of (X, y) of the current minibatch/chunk
                x_batch = np.array([x.flatten() for x in X_train[i:i + batch_size]])
                y_batch = np.array([y for y in y_train[i:i + batch_size]])        
                self.train_batch(x_batch, y_batch)
    
            train_log.append(np.mean(self.predict(X_train) == y_train.argmax(axis=-1)))                
            print(f"Epoch: {epoch + 1}, Train accuracy: {train_log[-1]}")                        
        return train_log
    
    def predict(self, X):
        # Compute network predictions. Returning indices of largest Logit probability
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)
    
    def predict_proba(self, X):
        # Return probability distribution over classes
        logits = self.forward(X)[-1]
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs
        
    def save_model(self, filepath):
        """Save the model to a file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
            
    @staticmethod
    def load_model(filepath):
        """Load a model from a file"""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

def normalize(X):
    X_normalize = (X - np.min(X)) / (np.max(X) - np.min(X))
    return X_normalize   

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class DigitDrawingApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Digit Classifier - Draw & Predict")
        self.model = model
        
        # Drawing parameters
        self.brush_size = 20
        self.color = "black"
        self.grid_size = 28
        self.cell_size = 15  # Size of each cell in pixels
        self.canvas_size = self.grid_size * self.cell_size
        
        # Drawing state
        self.drawing_grid = np.zeros((self.grid_size, self.grid_size))
        self.is_drawing = False
        self.eraser_mode = False
        
        # Create main frames
        self.left_frame = Frame(self.root)
        self.left_frame.pack(side="left", fill="both", expand=True)
        
        self.right_frame = Frame(self.root)
        self.right_frame.pack(side="right", fill="both", expand=True)
        
        # Create drawing canvas
        self.canvas = Canvas(self.left_frame, width=self.canvas_size, 
                             height=self.canvas_size, bg="white", bd=2, relief="ridge")
        self.canvas.pack(pady=10, padx=10)
        
        # Create drawing grid
        self.draw_grid()
        
        # Create buttons
        self.button_frame = Frame(self.left_frame)
        self.button_frame.pack(pady=10)
        
        self.pencil_button = Button(self.button_frame, text="Pencil", command=self.use_pencil)
        self.pencil_button.pack(side="left", padx=5)
        
        self.eraser_button = Button(self.button_frame, text="Eraser", command=self.use_eraser)
        self.eraser_button.pack(side="left", padx=5)
        
        self.clear_button = Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side="left", padx=5)
        
        self.predict_button = Button(self.left_frame, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)
        
        # Create prediction display
        self.prediction_label = Label(self.left_frame, text="Draw a digit and click 'Predict'", font=("Arial", 14))
        self.prediction_label.pack(pady=10)
        
        # Create probability frame
        self.prob_frame = Frame(self.left_frame)
        self.prob_frame.pack(pady=10)
        
        # Create labels for top 3 probabilities
        self.prob_labels = []
        for i in range(3):
            label = Label(self.prob_frame, text="", font=("Arial", 12))
            label.pack(pady=2)
            self.prob_labels.append(label)
        
        # Create network visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.canvas_plt = FigureCanvasTkAgg(self.fig, self.right_frame)
        self.canvas_plt.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Initialize with pencil mode
        self.use_pencil()
        
    def draw_grid(self):
        """Draw the grid lines on the canvas"""
        # Clear existing grid
        self.canvas.delete("grid")
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical lines
            self.canvas.create_line(
                i * self.cell_size, 0, 
                i * self.cell_size, self.canvas_size, 
                fill="#DDDDDD", tags="grid"
            )
            # Horizontal lines
            self.canvas.create_line(
                0, i * self.cell_size, 
                self.canvas_size, i * self.cell_size, 
                fill="#DDDDDD", tags="grid"
            )
    
    def use_pencil(self):
        self.eraser_mode = False
        self.pencil_button.config(relief="sunken")
        self.eraser_button.config(relief="raised")
    
    def use_eraser(self):
        self.eraser_mode = True
        self.pencil_button.config(relief="raised")
        self.eraser_button.config(relief="sunken")
    
    def start_draw(self, event):
        self.is_drawing = True
        self.draw(event)
    
    def draw(self, event):
        if not self.is_drawing:
            return
        
        # Get grid coordinates
        grid_x = event.x // self.cell_size
        grid_y = event.y // self.cell_size
        
        # Make sure we're within the grid
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            # Calculate cell coordinates
            x1 = grid_x * self.cell_size
            y1 = grid_y * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            
            # Update the grid state
            if self.eraser_mode:
                self.drawing_grid[grid_y, grid_x] = 0
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="#DDDDDD", tags="drawing")
            else:
                self.drawing_grid[grid_y, grid_x] = 1
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="#DDDDDD", tags="drawing")
    
    def stop_draw(self, event):
        self.is_drawing = False
    
    def clear_canvas(self):
        self.canvas.delete("drawing")
        self.drawing_grid = np.zeros((self.grid_size, self.grid_size))
        self.prediction_label.config(text="Draw a digit and click 'Predict'")
        for label in self.prob_labels:
            label.config(text="")
        self.fig.clf()
        self.canvas_plt.draw()
    
    def predict(self):
        # Normalize the image (0-1 scale)
        img = normalize(self.drawing_grid.flatten().reshape(1, -1))
        
        # Get prediction probabilities
        probs = self.model.predict_proba(img)[0]
        
        # Get the top prediction
        prediction = np.argmax(probs)
        
        # Update prediction label
        self.prediction_label.config(text=f"Prediction: {prediction}")
        
        # Get top 3 probabilities
        top_indices = np.argsort(probs)[::-1][:3]
        for i, (idx, prob_label) in enumerate(zip(top_indices, self.prob_labels)):
            prob_label.config(text=f"Digit {idx}: {probs[idx]:.4f} ({probs[idx]*100:.1f}%)")
        
        # Visualize network
        self.visualize_network(img)
    
    def visualize_network(self, input_data):
        # Clear the figure
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        
        # Get all layer activations
        activations = self.model.get_layer_activations(input_data)
        
        # Layer sizes (including input and output)
        layer_sizes = [input_data.shape[1]]  # Input layer
        
        # Add sizes of hidden layers and output layer
        current_layer_idx = 0
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer_sizes.append(layer.weights.shape[1])
                current_layer_idx += 1
            else:  # For non-Dense layers like ReLU
                current_layer_idx += 1
        
        # Calculate positions for nodes with special handling for output layer
        layer_positions = []
        
        # Get prediction probabilities for highlighting output layer
        probs = self.model.predict_proba(input_data)[0]
        predicted_digit = np.argmax(probs)
        
        for l, size in enumerate(layer_sizes):
            # Use different spacing strategy for each layer
            positions = []
            
            # Special handling for output layer (last layer)
            if l == len(layer_sizes) - 1:
                # Make output layer nodes larger and more spread out
                spacing = 13  # Larger spacing for output layer
                # Center the output layer vertically
                start_pos = -spacing * (size - 1) / 2
                for i in range(size):
                    positions.append(start_pos + i * spacing)
            else:
                # Scale vertical spacing based on layer size for other layers
                spacing = max(0.2, 12.0 / size)  # Smaller nodes = smaller spacing
                start_pos = -spacing * (size - 1) / 2
                for i in range(size):
                    positions.append(start_pos + i * spacing)
            
            layer_positions.append(positions)
        
        # Set lower threshold for connections to show more
        connection_threshold = 0.05  # Show connections > 5% of max weight
        
        # Draw the connections between layers
        for l in range(len(layer_sizes) - 1):
            layer_idx = l * 2  # Adjust for ReLU layers
            
            # For Dense layers, get weights
            if isinstance(self.model.layers[layer_idx], Dense):
                weights = self.model.layers[layer_idx].weights
                max_weight = np.max(np.abs(weights))
                
                # If connecting to output layer, show all connections
                # Otherwise, use sampling for very large layers
                if l == len(layer_sizes) - 2:  # Last hidden to output
                    # Show all connections to output layer
                    sample_i = range(len(layer_positions[l]))
                    sample_j = range(len(layer_positions[l+1]))
                else:
                    # For very large layers, use sampling to avoid excessive connections
                    stride_i = max(1, len(layer_positions[l]) // 100)
                    stride_j = max(1, len(layer_positions[l+1]) // 100)
                    sample_i = range(0, len(layer_positions[l]), stride_i)
                    sample_j = range(0, len(layer_positions[l+1]), stride_j)
                
                for i in sample_i:
                    for j in sample_j:
                        if i < weights.shape[0] and j < weights.shape[1]:
                            weight = weights[i, j]
                            if abs(weight) > connection_threshold * max_weight:
                                # Calculate color and width based on weight
                                color = 'red' if weight > 0 else 'blue'
                                width = 0.1 + 1.5 * abs(weight) / max_weight  # Thinner lines overall
                                alpha = 0.1 + 0.5 * abs(weight) / max_weight  # More transparent overall
                                
                                # Make connections to predicted digit more visible
                                if l == len(layer_sizes) - 2 and j == predicted_digit:
                                    width *= 2  # Double width for connections to predicted digit
                                    alpha = min(1.0, alpha * 2)  # More visible
                                
                                # Draw the line
                                self.ax.plot([l, l+1], 
                                          [layer_positions[l][i], layer_positions[l+1][j]], 
                                          color=color, linewidth=width, alpha=alpha)
        
        # Draw the nodes
        for l, positions in enumerate(layer_positions):
            # Get activations for this layer
            if l == 0:
                layer_activations = input_data[0].reshape(-1)  # Input layer
            else:
                layer_idx = l - 1  # Adjust for layer indexing
                layer_activations = activations[layer_idx][0]
            
            # Normalize activations to 0-1 for color mapping
            if np.max(layer_activations) > np.min(layer_activations):
                act_normalized = (layer_activations - np.min(layer_activations)) / (np.max(layer_activations) - np.min(layer_activations))
            else:
                act_normalized = np.zeros_like(layer_activations)
            
            # For very large layers, use sampling to avoid excessive nodes
            if len(positions) > 200:
                stride = len(positions) // 200
                sample_indices = range(0, len(positions), stride)
            else:
                sample_indices = range(len(positions))
            
            # Set node size based on layer
            if l == len(layer_sizes) - 1:  # Output layer
                node_size = 800  # Larger nodes for output layer
            else:
                node_size = max(20, 500 // len(positions))  # Scale node size inversely with layer size
            
            # Draw nodes
            for i in sample_indices:
                if i < len(layer_activations):
                    # Set node color based on activation
                    activation = act_normalized[i] if i < len(act_normalized) else 0
                    color = cm.viridis(activation)
                    
                    # Special handling for output layer
                    if l == len(layer_sizes) - 1:
                        # Use probability for node size in output layer
                        prob = probs[i]
                        size = node_size  #* (0.5 + 2.0 * prob)  # Size ranges from 50% to 250% based on probability
                        
                        # Highlight the predicted digit
                        if i == predicted_digit:
                            edgecolor = 'yellow'
                            edgewidth = 3
                        else:
                            edgecolor = 'black'
                            edgewidth = 1
                        
                        # Add text label for output nodes showing digit and probability
                        self.ax.text(l, positions[i], f"{i}\n{prob:.2f}", ha='center', va='center', 
                                  fontsize=8, fontweight='bold', color='white', zorder=15)
                    else:
                        size = node_size
                        edgecolor = 'black'
                        edgewidth = 1
                    
                    # Draw the node
                    self.ax.scatter(l, positions[i], s=size, color=color, 
                                 edgecolors=edgecolor, linewidth=edgewidth, zorder=10)
            
            # Add labels for layers
            if l == 0:
                self.ax.text(l, positions[0] - 4, "Input\nLayer", ha='center', fontsize=10)
            elif l == len(layer_positions) - 1:
                self.ax.text(l, positions[-1] + 4, "Output\nLayer", ha='center', fontsize=12, fontweight='bold')
            else:
                self.ax.text(l, positions[0] - 4, f"Hidden\nLayer {l}", ha='center', fontsize=10)
        
        # Set axis limits and remove ticks
        self.ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
        self.ax.axis('off')
        self.ax.set_title("Neural Network Visualization", fontsize=14)
        
        # Add colorbar for activation values
        sm = plt.cm.ScalarMappable(cmap=cm.viridis)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.ax, orientation='vertical', shrink=0.8)
        cbar.set_label('Activation Value')
        
        # Add legend for connection weights
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Positive Weight'),
            Line2D([0], [0], color='blue', lw=2, label='Negative Weight')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        # Display the plot
        self.canvas_plt.draw()

'''
def train_or_load_model(X_train, Y_train, X_test, Y_test, model_path='mnist_model.pkl', force_train=False):
    """Train a new model or load an existing one"""
    if not force_train and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        network = MCP.load_model(model_path)
        if network is None:
            print("Failed to load model, creating a new one...")
            network = create_and_train_model(X_train, Y_train, model_path)
    else:
        print("Training a new model...")
        network = create_and_train_model(X_train, Y_train, model_path)
    
    # Evaluate the model
    test_corrects = len(list(filter(lambda x: x == True, network.predict(X_test) == Y_test.argmax(axis=-1))))
    test_all = len(X_test)
    test_accuracy = test_corrects/test_all
    print(f"Test accuracy = {test_corrects}/{test_all} = {test_accuracy}")
    
    return network

def create_and_train_model(X_train, Y_train, model_path):
    """Create and train a new model"""
    input_size = X_train.shape[1]
    output_size = Y_train.shape[1]
    
    network = MCP()
    network.add_layer(Dense(input_size, 100, learning_rate=0.05))
    network.add_layer(ReLU())
    network.add_layer(Dense(100, 200, learning_rate=0.05))
    network.add_layer(ReLU())
    network.add_layer(Dense(200, output_size))
    
    train_log = network.train(X_train, Y_train, n_epochs=20, batch_size=64)
    
    # Save the trained model
    network.save_model(model_path)
    
    # Plot training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_log, label='Train Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_accuracy.png')
    plt.close()
    
    return network

def main():
    # Set file paths based on added MNIST Datasets
    training_images_filepath = './mnist-dataset/train-images.idx3-ubyte'
    training_labels_filepath = './mnist-dataset/train-labels.idx1-ubyte'
    test_images_filepath = './mnist-dataset/t10k-images.idx3-ubyte'
    test_labels_filepath = './mnist-dataset/t10k-labels.idx1-ubyte'
    
    # Load MNIST dataset
    print('Loading MNIST dataset...')
    mnist_dataloader = MnistDataLoader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    print('MNIST dataset loaded.')
    
    # Preprocess data
    X_train = normalize(np.array([np.ravel(x) for x in x_train]))
    X_test = normalize(np.array([np.ravel(x) for x in x_test]))
    Y_train = np.array([one_hot(np.array(y, dtype=int), 10) for y in y_train], dtype=int)
    Y_test = np.array([one_hot(np.array(y, dtype=int), 10) for y in y_test], dtype=int)
    
    # Check if user wants to force train a new model
    force_train = False
    response = input("Do you want to force training a new model? (y/n, default: n): ")
    if response.lower() == 'y':
        force_train = True
    
    # Train or load the model
    model = train_or_load_model(X_train, Y_train, X_test, Y_test, force_train=force_train)
    
    # Create drawing application
    root = tk.Tk()
    app = DigitDrawingApp(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()