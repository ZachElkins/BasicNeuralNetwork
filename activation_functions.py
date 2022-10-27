import numpy as np


activation_functions = {
    "sigmoid": {
        "base": lambda x: 1 / (1+(np.e**-x)),
        "deriv": lambda y: y*(1-y)
    },
    "tanh": {
        "base": lambda x: np.tanh(x),
        "deriv": lambda y: 1 - (y * y)
    },
    "relu": {
        "base": lambda x: np.maximum(0, x),
        "deriv": lambda y: np.greater(y, 0).astype('float32')
    }
}
