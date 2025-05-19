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

from models.mcp import MCP
from models.dense import Dense
from models.relu import ReLU
from models.layer import Layer

from utils.mnist_dataloader import *
from utils.math_utils import *

from gui.digit_drawing_app import DigitDrawingApp

# info: stochastic gradient descent model
# ReLU activation functoin
# binary cross-entropy
# 784 x 100 x 200 x 10 parameters
# dense architecture

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
    
    train_log = network.train(X_train, Y_train, n_epochs=60, batch_size=100)
    
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