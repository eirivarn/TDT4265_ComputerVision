import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer

def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    
    # Model with default settings (no improved sigmoid, no improved weight init)
    model_default = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid=False,
        use_improved_weight_init=False,
        use_relu=False)
    trainer_default = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model_default, 
        learning_rate, 
        batch_size, 
        shuffle_data,
        X_train, 
        Y_train, 
        X_val, 
        Y_val) 
    
    train_history_default, val_history_default = trainer_default.train(num_epochs)

    # Model with improved sigmoid and improved weight initialization
    model_improved = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid=True,
        use_improved_weight_init=True,
        use_relu=False)
    trainer_improved = SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model_improved, 
        learning_rate, 
        batch_size, 
        shuffle_data,
        X_train, 
        Y_train, 
        X_val, 
        Y_val)
    
    train_history_improved, val_history_improved = trainer_improved.train(num_epochs)

    # Plotting the comparison
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history_default["loss"], "Default Model", npoints_to_average=10)
    utils.plot_loss(train_history_improved["loss"], "Improved Model", npoints_to_average=10)
    plt.title("Training Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.ylim([0, .4])

    plt.subplot(1, 2, 2)
    utils.plot_loss(val_history_default["accuracy"], "Default Model")
    utils.plot_loss(val_history_improved["accuracy"], "Improved Model")
    plt.title("Validation Accuracy")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim([0.85, .95])
    plt.show()

if __name__ == "__main__":
    main()
