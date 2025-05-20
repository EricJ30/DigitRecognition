import numpy as np
from models.layer import Layer
class MaxPool2D(Layer):
    def __init__(self, size=2, stride=2):
        super().__init__()
        self.size = size
        self.stride = stride

    def forward(self, input):
        self.input = input
        batch_size, channels, in_h, in_w = input.shape
        pool_h = (in_h - self.size) // self.stride + 1
        pool_w = (in_w - self.size) // self.stride + 1
        output = np.zeros((batch_size, channels, pool_h, pool_w))

        self.max_indices = np.zeros_like(output, dtype=int)

        for i in range(pool_h):
            for j in range(pool_w):
                region = input[:, :, i*self.stride:i*self.stride+self.size, j*self.stride:j*self.stride+self.size]
                output[:, :, i, j] = np.max(region, axis=(2,3))
        return output

    def backward(self, input, grad_output):
        batch_size, channels, in_h, in_w = input.shape
        grad_input = np.zeros_like(input)
        pool_h, pool_w = grad_output.shape[2:]

        for i in range(pool_h):
            for j in range(pool_w):
                region = input[:, :, i*self.stride:i*self.stride+self.size, j*self.stride:j*self.stride+self.size]
                max_vals = np.max(region, axis=(2,3), keepdims=True)
                mask = (region == max_vals)
                grad_input[:, :, i*self.stride:i*self.stride+self.size, j*self.stride:j*self.stride+self.size] += \
                    grad_output[:, :, i:i+1, j:j+1] * mask
        return grad_input
