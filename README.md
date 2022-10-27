# Basic Neural Network
*A simple python implementation of a neural network for learning about the subject.*

#### The project is made up of five important files

  * `train.py` — Create and train a neural network
  * `test.py` — Test the error rate of a neural network created with `train.py`
  * `neuralnetwork.py` — The `NeuralNetwork` class implementation.
  * `activation_functions.py` — A dict with the different activation functions, and their derivatives, that can be used in the nueral networks.
  * `utils.py` — Miscellaneous functions utilized by other processes.

#### `Train.py` and `Test.py` have optional command line arguments listed below

**train.py**

```
usage: train.py [-h] [-hn HN] [-a A] [-lr LR] [-t T] [-g] [-n N]

Train a Neural Network on the mnist dataset.

optional arguments:
  -h, --help  show this help message and exit
  -hn HN      The number of hidden nodes to use. (Default: 700)
  -a A        The activation function to use. Sigmoid, Tanh, ReLU. (Default: Sigmoid)
  -lr LR      The learning rate. Must be within the exclusive range (0, 1). (Default: 0.05)
  -t T        The number of images used to train the model in the range (0, 60000]. (Default: 30000)
  -g          Flag to graph the training over time. (This will add significant time to training)
  -n N        Filename, not including extension. (Default: model)
```

**test.py**
```
usage: test.py [-h] [-t T] [-f F]

Test a Neural Network created with train.py

optional arguments:
  -h, --help  show this help message and exit
  -t T        The number of images used to test the model in the range (0, 10000]. (Default: 1000)
  -f F        Filename of model to test, not including extension. (Default: model)
```