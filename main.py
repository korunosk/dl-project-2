import torch
import src.dataset as dataset
import src.nn as nn
import src.optim as optim
import src.viz as viz

torch.set_grad_enabled(False)

# Depending on the initialization, the solution can get stuck
# in a local optimum. Re-execute the code if that is the case.

if __name__ == '__main__':
    # Globals
    N = 10000
    batch_size = 200
    epochs = 10

    # Generate train and test datasets
    X_train, y_train = dataset.gen_dataset(N)
    X_test, y_test = dataset.gen_dataset(N)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Define module
    model = nn.Sequential(
        nn.Linear(2, 25),
        nn.Relu(),
        nn.Linear(25, 25),
        nn.Relu(),
        nn.Linear(25, 25),
        nn.Relu(),
        nn.Linear(25, 2),
    )

    # Initialize the optimizer
    optim = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Initilize the loss
    criterion = nn.LossMSE()

    # Accumulator arrays for the loss and the accuracy
    loss = []
    accuracy = []

    # Iterate over the dataset in several epochs
    for epoch in range(epochs):

        # For each epoch, calculate the accuracy of the model
        accuracy.append([epoch * N, 0])

        # For each batch in the test dataset...
        for i in range(0, N, batch_size):
            input = X_test[i:i+batch_size]
            target = y_test[i:i+batch_size]

            # Calculate the accuracy...
            prediction = model.forward(input) > 0.5

            # And update the accumulator array for that epoch
            accuracy[-1][1] += ((prediction == target).sum(dim=0).float() / N)[0].item()

        # Print the accuracy
        print(f'Accuracy: {100 * accuracy[-1][1]:>6.2f}')

        if epoch == epochs - 1:
            break

        # For each batch in the train dataset...
        for i in range(0, N, batch_size):
            
            loss.append([epoch * N + i, 0])
            
            input = X_train[i:i+batch_size]
            target = y_train[i:i+batch_size]

            # Execute the forward passs...
            prediction = model.forward(input)

            # Calculate the loss...
            loss[-1][1] = criterion.forward(target, prediction)[0].item()
            
            # And print the loss...
            print(f'Loss: {loss[-1][1]:6.2f}')

            # Then, execute the backward passs
            model.backward(criterion.backward())

            # Update the parameters
            optim.step()
            # Zero out the gradients
            optim.zero_grad()

    # Plot the loss vs. accuracy and the decision boundary
    viz.plot_loss_accuracy(loss, accuracy)
    viz.plot_decision_bountary(model, X_test, y_test)
