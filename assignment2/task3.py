import utils
import matplotlib.pyplot as plt
from task2a import  cross_entropy_loss, pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer, calculate_accuracy

def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # Initialize models and trainers for each configuration
    configurations = [
        ('Default Model', False, False, False),
        ('Improved Weight', True, False, False),
        ('Improved Sigmoid', False, True, False),
        ('Improved Weight and Sigmoid', True, True, False),
        ('Default Model With Momentum', False, False, True),
        ('Complete Improved Model', True, True, True),
    ]
    
    for label, use_improved_weight_init, use_improved_sigmoid, use_momentum in configurations:
        if use_momentum:
            learning_rate = .02
        # Initialize the model and trainer
        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init,
            use_relu=False)
        trainer = SoftmaxTrainer(
            momentum_gamma,
            use_momentum,
            model,
            learning_rate,
            batch_size,
            shuffle_data,
            X_train,
            Y_train,
            X_val,
            Y_val)
        
        print("Training new model")
        print("Use Improved Weight Init:", use_improved_weight_init)
        print("Use Improved Sigmoid:", use_improved_sigmoid)
        print("Use Momentum:", use_momentum)
        print()

        # Train the model
        train_history, val_history = trainer.train(num_epochs)
    
        print(
        "Final Train Cross Entropy Loss:",
        cross_entropy_loss(Y_train, model.forward(X_train)),
        )
        print(
            "Final Validation Cross Entropy Loss:",
            cross_entropy_loss(Y_val, model.forward(X_val)),
        )
        print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
        print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

        # Plot training loss and validation accuracy
        plt.subplot(1, 2, 1)
        utils.plot_loss(train_history["loss"], label, npoints_to_average=10)
        plt.title("Training Loss")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.ylim([0, .4])

        plt.subplot(1, 2, 2)
        utils.plot_loss(val_history["accuracy"], label)
        plt.title("Validation Accuracy")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.ylim([0.75, 1.0])

    # Save the plot
    plt.savefig("assignment2/images/task3_complete.png")
    # Show the plot
    plt.show()
    
    model_configurations = [
        ('1 Layer - 64 Nodes', [64, 10]),
        ('2 Layers - 64 Nodes each', [64, 64, 10]),
        ('1 Layer - 100 Nodes', [100, 10]),
        ('2 Layers - 100 and 30 Nodes', [100, 30, 10]),
        # Add more configurations as needed
    ]

    # Initialize models and trainers for each configuration
    for label, neurons_per_layer in model_configurations:
        # Initialize the model and trainer
        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid=False,  
            use_improved_weight_init=False,  
            use_relu=False)  
        trainer = SoftmaxTrainer(
            momentum_gamma,
            use_momentum,  
            model,
            learning_rate,
            batch_size,
            shuffle_data,
            X_train,
            Y_train,
            X_val,
            Y_val)

        print(f"Training model: {label}")
        
        # Train the model
        train_history, val_history = trainer.train(num_epochs)
    
        print(f"Final Train Cross Entropy Loss for {label}:",
              cross_entropy_loss(Y_train, model.forward(X_train)))
        print(f"Final Validation Cross Entropy Loss for {label}:",
              cross_entropy_loss(Y_val, model.forward(X_val)))
        print(f"Train accuracy for {label}:", calculate_accuracy(X_train, Y_train, model))
        print(f"Validation accuracy for {label}:", calculate_accuracy(X_val, Y_val, model))

        # Plot training loss and validation accuracy
        plt.subplot(1, 2, 1)
        utils.plot_loss(train_history["loss"], label, npoints_to_average=10)
        plt.title("Training Loss")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.ylim([0, .4])

        plt.subplot(1, 2, 2)
        utils.plot_loss(val_history["accuracy"], label)
        plt.title("Validation Accuracy")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.ylim([0.75, 1.0])

    # Save the plot
    plt.savefig("assignment2/images/task4_model_topology_comparisons.png")
    # Show the plot
    plt.show()
    
if __name__ == "__main__":
    main()
