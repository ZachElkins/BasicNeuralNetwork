from ast import Raise

import numpy as np

from activation_functions import activation_functions as active_fns


class NeuralNetwork:
 
    def __init__(
        self,
        input_nodes,
        hidden_nodes,
        output_nodes,
        activation_function,
        learning_rate = 0.1
    ):
        if learning_rate <= 0 or learning_rate >= 1:
            raise ValueError("Learning rate must be within the range (0, 1)")
        
        if activation_function not in active_fns.keys():
            raise ValueError(f"Activation function must be in {active_fns.keys()}")

        active_fns[activation_function]
        # Weights are intialized to a list of random numbers, then training refines those numbers too give good results
        self.w_ih = 2*np.random.rand(hidden_nodes, input_nodes)-1 # Weights from the input to hidden layer
        self.w_ho = 2*np.random.rand(output_nodes, hidden_nodes)-1 # Weights from the hidden to output layer
        self.bias_h = 2*np.random.rand(hidden_nodes, 1)-1 # Bias for the hidden layer
        self.bias_o = 2*np.random.rand(output_nodes, 1)-1 # Bias for the output layer
        self.learning_rate = learning_rate # Rate of gradient adjustment for back-propogation
        self.active_fn = activation_function # Activation function name
        
    def train(self, inputs, targets):
        '''
            Given an input, make a prediction. Then based off of the target calculate error
            and update weights through back-propogation
        '''
        # For each layer transition, calculated the weighted sum, then pass it through the activation function
        
        inputs = np.reshape(inputs, (len(inputs), 1))
        targets = np.reshape(targets, (len(targets), 1))
        # Hidden Layer
        hidden_inputs = np.dot(self.w_ih, inputs) # Inputs to the hidden layer are the weighted values of the input layer
        hidden_inputs = np.add(hidden_inputs, self.bias_h)
        hidden_outputs = np.vectorize(active_fns[self.active_fn]["base"])(hidden_inputs) # Output of the hidden layer are the input weights passed through the activation function
        # Output Layer
        output_inputs = np.dot(self.w_ho, hidden_outputs) # The inputs to the output layer
        output_inputs = np.add(output_inputs, self.bias_o)
        outputs = np.vectorize(active_fns[self.active_fn]["base"])(output_inputs)
        
        # Back-Propogation
        
        # Calculate error (Figure out and apply a percent change to the weights at each layer)
        output_err = targets - outputs
        hidden_err = np.dot(np.transpose(self.w_ho), output_err)
        # Create gradients (How the output should change given the errors)
        output_gradient = np.vectorize(active_fns[self.active_fn]["deriv"])(outputs) * self.learning_rate
        hidden_gradient = np.vectorize(active_fns[self.active_fn]["deriv"])(hidden_outputs) * hidden_err * self.learning_rate
        # Update hidden -> output weights
        w_ho_delta = np.dot(output_gradient, np.transpose(hidden_outputs))
        self.w_ho = self.w_ho+w_ho_delta
        # Update input -> hidden weights
        w_ih_delta = np.dot(hidden_gradient, np.transpose(inputs))
        self.w_ih = self.w_ih+w_ih_delta
        
    def predict(self, inputs):
        '''Make a prediction based off of a given input and return the resulting guess.'''
        inputs = np.reshape(inputs, (len(inputs), 1))
        
        hidden_inputs = np.dot(self.w_ih, inputs)
        hidden_inputs = np.add(hidden_inputs, self.bias_h)
        hidden_outputs = np.vectorize(active_fns[self.active_fn]["base"])(hidden_inputs)
        
        output_inputs = np.dot(self.w_ho, hidden_outputs)
        output_inputs = np.add(output_inputs, self.bias_o)
        outputs = np.vectorize(active_fns[self.active_fn]["base"])(output_inputs)
        
        return outputs

    def to_serializable(self):
        return {
            "ih": self.w_ih.astype(float).tolist(),
            "ho": self.w_ho.astype(float).tolist(),
            "bh": self.bias_h.astype(float).tolist(),
            "bo": self.bias_o.astype(float).tolist(),
            "fn": self.active_fn.lower(),
            "lr": self.learning_rate
        }

    @staticmethod
    def from_dict(data):
        new_nn = NeuralNetwork(
            input_nodes=0, output_nodes=0, hidden_nodes=0,
            activation_function=data["fn"].lower(), learning_rate=data["lr"]
        )
        new_nn.w_ih = np.array(data["ih"])        
        new_nn.w_ho = np.array(data["ho"])
        new_nn.bias_h = np.array(data["bh"])
        new_nn.bias_o = np.array(data["bo"])

        return new_nn