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
import sys
from models.mcp import MCP  # or whatever class/function is in mcp.py
from models.dense import Dense
from models.relu import ReLU
from models.layer import Layer
from scipy.ndimage import gaussian_filter
from utils.mnist_dataloader import *
from utils.math_utils import *
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
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
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
        # Get the original binary drawing
        original_img = self.drawing_grid.flatten().reshape(1, -1)
        
        # Apply Gaussian blur to the image (reshape to 28x28 for spatial operations)
        img_2d = original_img.reshape(28, 28)
        
        # Sigma controls blur intensity - adjust this parameter as needed
        sigma = 1.0
        blurred_img_2d = gaussian_filter(img_2d, sigma=sigma)
        
        # Reshape back to model input format and normalize
        img = normalize(blurred_img_2d.flatten().reshape(1, -1))
        img = img.reshape(-1, 1, 28, 28)
        
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
        
        # Optional: Display both original and blurred image for comparison
        # This requires you to have an additional canvas/display area in your UI
        # If you want to add this, you'll need to implement a method to update that display
        
        # Visualize network
        self.visualize_network(img)
    
    def visualize_network(self, input_data):
        # Clear the figure
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        
        # Get all layer activations
        activations = self.model.get_layer_activations(input_data)
        
        # Identify and collect only Dense layers (plus input and output)
        dense_layers = []
        dense_layer_indices = []
        
        # Add input layer
        dense_layers.append({"type": "input", "size": input_data.shape[1]})
        
        # Scan through model layers to find Dense layers
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, Dense):
                dense_layers.append({
                    "type": "dense", 
                    "size": layer.weights.shape[1],
                    "weights": layer.weights,
                    "index": i
                })
                dense_layer_indices.append(i)
        
        # Extract layer sizes for visualization (including input layer)
        layer_sizes = [layer["size"] for layer in dense_layers]
        
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
                spacing = 10  # Larger spacing for output layer
                # Center the output layer vertically
                start_pos = -spacing * (size - 1) / 2
                for i in range(size):
                    positions.append(start_pos + i * spacing)
            else:
                # Scale vertical spacing based on layer size for other layers
                spacing = max(0.6, 12.0 / size)  # Smaller nodes = smaller spacing
                start_pos = -spacing * (size - 1) / 2
                for i in range(size):
                    positions.append(start_pos + i * spacing)
            
            layer_positions.append(positions)
        
        # Set lower threshold for connections to show more
        connection_threshold = 0.05  # Show connections > 5% of max weight
        
        # Draw the connections between layers
        for l in range(len(dense_layers) - 1):
            # For Dense layers, get weights (skip input to first hidden if that first hidden is not Dense)
            if l > 0:  # Not the input layer
                weights = dense_layers[l]["weights"]
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
                # Find the corresponding activation based on the actual layer index
                layer_idx = dense_layer_indices[l-1]  # Adjust for layer indexing
                
                # Make sure we're accessing a valid index in the activations list
                if layer_idx < len(activations):
                    layer_activations = activations[layer_idx][0]
                else:
                    # Fallback if activations aren't available
                    layer_activations = np.zeros(dense_layers[l]["size"])
            
            # Normalize activations to 0-1 for color mapping
            if len(layer_activations) > 0 and np.max(layer_activations) > np.min(layer_activations):
                act_normalized = (layer_activations - np.min(layer_activations)) / (np.max(layer_activations) - np.min(layer_activations))
            else:
                act_normalized = np.zeros(dense_layers[l]["size"])
            
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
                        prob = probs[i] if i < len(probs) else 0
                        size = node_size  # Size previously scaled by probability - now just fixed
                        
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
                self.ax.text(l, positions[0] - 12, "Input\nLayer", ha='center', fontsize=12, fontweight='bold')
            elif l == len(layer_positions) - 1:
                self.ax.text(l, positions[-1] + 4, "Output\nLayer", ha='center', fontsize=12, fontweight='bold')
            else:
                # Determine the actual layer number based on the original layer indices
                if l == 1:  # First hidden layer 
                    layer_name = f"Dense Layer 1\n(After CNN layers)"
                else:
                    layer_name = f"Dense Layer {l}"
                self.ax.text(l, positions[0] - 12, layer_name, ha='center', fontsize=12, fontweight='bold')
        
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
    def on_closing(self):
        self.root.quit()
        self.root.destroy()