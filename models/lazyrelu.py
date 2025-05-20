import numpy as np
from models.layer import Layer

class LazyReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, input):
        self.mask = input > 0
        return input * self.mask

    def backward(self, input, grad_output):
        return grad_output * self.mask
