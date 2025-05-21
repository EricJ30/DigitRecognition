import numpy as np
import pickle
from models.layer import Layer
from utils.math_utils import *
import time

# Multi-layer perceptron neural network
class MCP:
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
    
    """Compute activations of all network layers by applying them sequentially.
        Return a list of activations for each layer."""
    def forward(self, X):
        
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
        
        layer_activations = self.forward(X)
        layer_inputs = [X] + layer_activations  # layer_input[i] is an input for layer[i]
        logits = layer_activations[-1]
        
        # Compute the loss and the initial gradient    
        y_argmax = y.argmax(axis=1)        
        loss = softmax_crossentropy_with_logits(logits, y_argmax)
        loss_grad = grad_softmax_crossentropy_with_logits(logits, y_argmax)
    
        # Propagate gradients through the network
        # Reverse propagation as this is backprop
        for layer_index in range(len(self.layers))[::-1]:
            layer = self.layers[layer_index]        
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)  # grad w.r.t. input, also weight updates
        
        return np.mean(loss)
    
    def train(self, X_train, y_train, n_epochs=20, batch_size=8):
        train_log = []        
        # Before feeding to network

        for epoch in range(n_epochs):     
            start = time.time()   
            for i in range(0, X_train.shape[0], batch_size):
                # Get pair of (X, y) of the current minibatch/chunk
                if len(X_train.shape) == 4:  # CNN input
                    x_batch = X_train[i:i + batch_size].reshape(-1, 1, 28, 28)
                else:  # Dense input
                    x_batch = np.array([x.flatten() for x in X_train[i:i + batch_size]])

                y_batch = np.array([y for y in y_train[i:i + batch_size]])        
                self.train_batch(x_batch, y_batch)
    
            train_log.append(np.mean(self.predict(X_train) == y_train.argmax(axis=-1)))                
            print(f"Epoch: {epoch + 1}, Train accuracy: {train_log[-1]}")   
            print(f"Batch took {time.time() - start:.2f}s")                     
        return train_log
    #Compute network predictions. Returning indices of largest Logit probability
    def predict(self, X):
        
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)
    
    #Return probability distribution over classes
    def predict_proba(self, X):

        logits = self.forward(X)[-1]
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs
        
    def save_model(self, filepath):
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