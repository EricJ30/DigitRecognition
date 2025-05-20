from models.layer import Layer
class Flatten(Layer):
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, input, grad_output):
        return grad_output.reshape(self.input_shape)
