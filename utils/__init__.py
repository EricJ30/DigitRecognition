# Make key functions available directly from the utils package
from .math_utils import *
from .mnist_dataloader import MnistDataLoader

# This allows usage like:
# from utils import softmax, load_mnist