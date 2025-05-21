import numpy as np
from models.layer import Layer

"""
input_data: (batch, channels, height, width)
Returns a 2D matrix (col) where each row is a flattened receptive field
"""
def im2col(input_data, kernel_size, stride, padding):

    N, C, H, W = input_data.shape
    KH, KW = kernel_size, kernel_size
    out_h = (H + 2 * padding - KH) // stride + 1
    out_w = (W + 2 * padding - KW) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (padding, padding), (padding, padding)], mode='constant')
    col = np.zeros((N, C, KH, KW, out_h, out_w))

    for y in range(KH):
        y_max = y + stride * out_h
        for x in range(KW):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col, out_h, out_w

"""
Converts 2D matrix back to 4D image (for gradient propagation)
"""
def col2im(col, input_shape, kernel_size, stride, padding):
    
    N, C, H, W = input_shape
    KH, KW = kernel_size, kernel_size
    out_h = (H + 2 * padding - KH) // stride + 1
    out_w = (W + 2 * padding - KW) // stride + 1

    col = col.reshape(N, out_h, out_w, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))

    for y in range(KH):
        y_max = y + stride * out_h
        for x in range(KW):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, padding:H + padding, padding:W + padding]

class Conv2D(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, learning_rate=0.01):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate

        scale = np.sqrt(2 / (input_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * scale
        self.biases = np.zeros(output_channels)

    def forward(self, input):
        self.input = input
        self.batch_size = input.shape[0]

        self.cols, self.out_h, self.out_w = im2col(
            input, self.kernel_size, self.stride, self.padding
        )
        W_col = self.weights.reshape(self.output_channels, -1)

        out = np.dot(self.cols, W_col.T) + self.biases
        out = out.reshape(self.batch_size, self.out_h, self.out_w, self.output_channels)
        return out.transpose(0, 3, 1, 2)  # (batch, channels, height, width)

    def backward(self, input, grad_output):
        grad_output = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.output_channels)

        W_col = self.weights.reshape(self.output_channels, -1)
        grad_weights = np.dot(grad_output.T, self.cols)
        grad_biases = grad_output.sum(axis=0)

        grad_input_col = np.dot(grad_output, W_col)
        grad_input = col2im(
            grad_input_col, self.input.shape, self.kernel_size, self.stride, self.padding
        )

        # Update weights
        self.weights -= self.learning_rate * grad_weights.reshape(self.weights.shape)
        self.biases -= self.learning_rate * grad_biases

        return grad_input
