
import math


def perceptron(x):
    return 1 if x > 0 else 0


def sigmoid(x):
    return 1. / (1 + math.exp(-x))
