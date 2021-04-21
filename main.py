import torch
import dataset
import nn
import optim
import viz

torch.set_grad_enabled(False)


if __name__ == '__main__':
    N = 10000
    batch_size = 200
    epochs = 10

    X_train, y_train = dataset.gen_dataset(N)
    X_test, y_test = dataset.gen_dataset(N)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    model = nn.Sequential(
        nn.Linear(2, 25),
        nn.Relu(),
        nn.Linear(25, 25),
        nn.Relu(),
        nn.Linear(25, 25),
        nn.Relu(),
        nn.Linear(25, 2),
    )

    optim = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    criterion = nn.LossMSE()

    loss = []
    accuracy = []

    for epoch in range(epochs):

        accuracy.append([epoch * N, 0])

        for i in range(0, N, batch_size):
            input = X_test[i:i+batch_size]
            target = y_test[i:i+batch_size]

            prediction = model.forward(input) > 0.5

            accuracy[-1][1] += ((prediction == target).sum(dim=0).float() / N)[0].item()

        print(f'Accuracy: {100 * accuracy[-1][1]:>6.2f}')

        if epoch == epochs - 1:
            break

        for i in range(0, N, batch_size):
            loss.append([epoch * N + i, 0])

            input = X_train[i:i+batch_size]
            target = y_train[i:i+batch_size]

            prediction = model.forward(input)

            loss[-1][1] = criterion.forward(target, prediction)[0].item()
            
            print(f'Loss: {loss[-1][1]:6.2f}')

            model.backward(criterion.backward())

            optim.step()
            optim.zero_grad()
    
    viz.plot_loss_accuracy(loss, accuracy)
    viz.plot_decision_bountary(model, X_test, y_test)
