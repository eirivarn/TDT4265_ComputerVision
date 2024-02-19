import numpy as np
import utils
import typing

np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"

    # Normalize the images
    X_avg = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_avg) / (X_std + 1e-6)

    # Apply bias trick
    bias_term = np.ones((X.shape[0], 1))
    X = np.concatenate((X, bias_term), axis=1)
    
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert (
        targets.shape == outputs.shape
    ), f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    cross_entropy_loss = - np.sum(targets * np.log(outputs))/len(targets)
    return cross_entropy_loss


class SoftmaxModel:

    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3b hyperparameter
        use_improved_weight_init: bool,  # Task 3a hyperparameter
        use_relu: bool,  # Task 3c hyperparameter
    ):
        np.random.seed(
            1
        )  # Always reset random seed before weight init to get comparable results.
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        self.ws = []  # List to store weight matrices for each layer
        previous_layer_size = self.I  # Number of nodes in the previous layer (initially set to input layer size)

        # Iterate through each layer defined in neurons_per_layer to initialize weights
        for current_layer_size in self.neurons_per_layer:
            # Define the shape of the weight matrix for the current layer
            weight_matrix_shape = (previous_layer_size, current_layer_size)
            print("Initializing weights with shape:", weight_matrix_shape)
            
            # Initialize weight matrix with random values
            if self.use_improved_weight_init:
                # Improved weight initialization method
                weight_matrix = np.random.normal(0, 1/np.sqrt(previous_layer_size), weight_matrix_shape)
            else:
                # Standard weight initialization (uniform distribution)
                weight_matrix = np.random.uniform(-1, 1, weight_matrix_shape)
            self.ws.append(weight_matrix)
            previous_layer_size = current_layer_size
            
        self.gradients = [None for _ in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        
        # Hidden layer
        hidden_weights = self.ws[0]
        z_hidden = np.dot(X, hidden_weights)
        self.hidden_activation = 1 / (1 + np.exp(-z_hidden)) #This is essentially the hidden neurons
            
        #output layer
        output_weights = self.ws[1]
        z_output = np.dot(self.hidden_activation, output_weights)
        predictions = np.exp(z_output) / np.sum(np.exp(z_output), axis=1, keepdims=True)
        
        return predictions
        

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []
        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

        # Calculate the error
        output_error = -(targets - outputs)
        grad_output = np.dot(self.hidden_activation.T, output_error) / len(X)

        hidden_error = np.dot(output_error, self.ws[1].T) * self.hidden_activation * (1 - self.hidden_activation)
        grad_hidden = np.dot(X.T, hidden_error) / len(X)

        self.grads = [grad_hidden, grad_output]


    def zero_grad(self) -> None:
        self.grads = [np.zeros_like(w) for w in self.ws]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    num_examples = Y.shape[0]
    one_hot_encoded = np.zeros((num_examples, num_classes))

    for i in range(num_examples):
        label = Y[i]
        one_hot_encoded[i, label] = 1
    return one_hot_encoded


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edidted.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
