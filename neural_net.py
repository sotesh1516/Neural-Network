from typing import Sequence
import numpy as np

class MyNeuralNetwork:
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int
    ):
        """This initializes the model.
        Weights and biases are in self.params which has the form:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension of input
            hidden_size: List with the number of neurons per hidden layer
            output_size: output dimension C
            num_layers: Number of layers
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)  #hidden_size looks like [3,4,5] where each index value is the number of neurons per hidden layer
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            # this is initializing the weights
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected layer.
        Parameters:
            W: the weights
            X: the input data
            b: the bias
        Returns:
            the output
        """
        output = np.dot(X, W)
        return output + b
    
    def linear_gradient(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg, N) -> np.ndarray:
        """Gradient of linear layer
            returns de_dw, de_db, de_dx (what's this called?)  
        """
        de_dx = np.dot(de_dz, np.transpose(W))
        de_dw = np.dot(np.transpose(X), de_dz)
        de_db = np.sum(de_dz, axis=0)
        return de_dx, de_dw, de_db

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        return np.maximum(0, X)

    def relu_gradient(self, X: np.ndarray) -> np.ndarray:
        """Gradient of ReLU.
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # new_arr = np.empty_like(X)
        # for i in range(len(X)):
        #     if X[i] > 0:
        #         new_arr[i] = 1
        #     else:
        #         new_arr[i] = 0
        # return new_arr
        return (X > 0).astype(float)

    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def sigmoid_gradient(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(X) * (1 - self.sigmoid(X))


    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.mean((p-y)**2)
    
    def mse_gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (2 * (p - y))/ len(y)
    
    def mse_sigmoid_gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.mse_gradient(y, p) * (p * (1-p))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters:
            X: Input data.
        Returns:
            Matrix of shape (N, C) (number of samples x output dim)
        """
        self.outputs = {}

        for i in range(1, self.num_layers+1):
            if i != self.num_layers:
                if i == 1:
                    self.outputs["input" + str(i)] = X
                else:
                    self.outputs["input" + str(i)] = self.outputs["post_act_output" + str(i-1)]
                self.outputs["pre_act_output" + str(i)] = pre_act = self.linear(self.params["W" + str(i)], self.outputs["input" + str(i)], self.params["b" + str(i)])
                self.outputs["post_act_output" + str(i)] = self.relu(pre_act)
            else:
                self.outputs["input" + str(i)] = self.outputs["post_act_output" + str(i-1)]
                self.outputs["pre_act_output" + str(i)] = pre_act = self.linear(self.params["W" + str(i)], self.outputs["input" + str(i)], self.params["b" + str(i)])
                self.outputs["post_act_output" + str(i)] = self.sigmoid(pre_act)
        return self.outputs["post_act_output" + str(self.num_layers)]

    def backward(self, y: np.ndarray) -> float:
        """Perform backprop and compute losses.
        Parameters:
            y: target values
        Returns:
            Loss for this batch
        """
        self.gradients = {}
        mse_loss = self.mse(y, self.outputs["post_act_output" + str(self.num_layers)])
        
        de_dz = self.mse_sigmoid_gradient(y, self.outputs["post_act_output" + str(self.num_layers)])
        for i in range(self.num_layers, 0, -1):
            
            current_input = self.outputs["input" + str(i)]
            
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]
            
            de_dx, de_dw, de_db = self.linear_gradient(W, current_input, b, de_dz, 0, len(y))
            
            self.gradients["dW" + str(i)] = de_dw
            self.gradients["db" + str(i)] = de_db
            
            if i > 1:
                prev_pre_act = self.outputs["pre_act_output" + str(i-1)]
                de_dz = de_dx * self.relu_gradient(prev_pre_act)
            
                
        
        return mse_loss

    def update(
        self,
        lr: float = 0 
    ):
        """Update the model parameters using SGD 
        Parameters:
            lr: Learning rate
        """
        for i in range(self.num_layers, 0, -1):
            self.params["W" + str(i)] -= lr * self.gradients["dW" + str(i)]
            self.params["b" + str(i)] -= lr * self.gradients["db" + str(i)]
