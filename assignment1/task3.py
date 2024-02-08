import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO Implement this function (Task 3b)
    outputs = model.forward(X)
    predicted_labels = np.argmax(outputs, axis=1)
    target_labels = np.argmax(targets, axis=1)
    accuracy = np.mean(predicted_labels == target_labels)
    return accuracy


def visualize_weights(model, title):
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i, ax in enumerate(axes):
        weight = model.w[:-1, i].reshape(28, 28)
        ax.imshow(weight, cmap='viridis')
        ax.set_title(f'Digit {i}')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        predictions = self.model.forward(X_batch)

        # Compute the loss between the predictions and the actual labels
        loss = cross_entropy_loss(Y_batch, predictions)

        # Backward pass: compute the gradient of the loss with respect to the model's weights
        self.model.backward(X_batch, predictions, Y_batch)

        # Update the weights using the computed gradients
        self.model.w -= self.model.grad * self.learning_rate

        # Return the computed loss
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    l2_lambdas = [1.0, 0.1, 0.01, 0.001]
    l2_norms = []
    validation_accuracies = {l2: [] for l2 in l2_lambdas}
    training_steps = []

    for l2_reg_lambda in l2_lambdas:
        print(f"Training with λ = {l2_reg_lambda}")

        # Initialize model
        model = SoftmaxModel(l2_reg_lambda)
        # Train model
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)
        validation_accuracies[l2_reg_lambda] = val_history['accuracy']

        if not training_steps:  # Assumes all training runs go for the same number of steps
            training_steps = [i for i in range(
                len(validation_accuracies[l2_reg_lambda]))]

        visualize_weights(
            model, f"Weights with L2 Regularization (λ = {l2_reg_lambda})")

        # Compute the L2 norm of the weights
        l2_norm = np.linalg.norm(model.w)
        l2_norms.append(l2_norm)

    plt.figure()
    plt.plot(l2_lambdas, l2_norms, marker='o')
    plt.xscale('log')
    plt.xlabel('λ value')
    plt.ylabel('L2 norm of weights')
    plt.title('L2 Norm of Weights for different λ values')
    plt.savefig("task4c_l2_norm_vs_lambda.png")
    plt.show()


if __name__ == "__main__":
    main()
