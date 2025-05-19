import numpy as np

class Layer:
    """Base layer class"""
    def __init__(self):        
        pass
    
    def forward(self, input):
        """Forward pass: receive input and return output"""
        return input
    
    def backward(self, input, grad_output):
        """Backward pass: receive gradient from next layer and return gradient for previous layer"""
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)