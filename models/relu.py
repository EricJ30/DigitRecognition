import numpy as np
from models.layer import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()
    # Apply elementwise ReLU to [batch, input_units] matrix
    def forward(self, input):
        relu_forward = np.maximum(0, input)
        return relu_forward
    
    # Compute gradient of loss w.r.t. ReLU input
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad