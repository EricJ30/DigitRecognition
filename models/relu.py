import numpy as np
from models.layer import Layer

class ReLU(Layer):
    """ReLU activation layer"""
    def __init__(self):
        # ReLU layer simply applies elementwise rectified linear unit to all inputs
        super().__init__()
    
    def forward(self, input):
        # Apply elementwise ReLU to [batch, input_units] matrix
        relu_forward = np.maximum(0, input)
        return relu_forward
    
    def backward(self, input, grad_output):
        # Compute gradient of loss w.r.t. ReLU input
        relu_grad = input > 0
        return grad_output * relu_grad