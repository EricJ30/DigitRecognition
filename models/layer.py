import numpy as np

class Layer:
    def __init__(self):        
        pass
    
    def forward(self, input):
        return input
    # Receive gradient from next layer and return gradient for previous layer
    def backward(self, input, grad_output):

        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)