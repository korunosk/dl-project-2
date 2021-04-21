import torch
from matplotlib import pyplot as plt


def plot_loss_accuracy(loss, accuracy):
    fig, ax = plt.subplots()

    x, y = zip(*loss)
    ax.plot(x, y, label='loss (train)')
   
    x,  y = zip(*accuracy)
    ax.plot(x, y, '-x', label='accuracy (test)')
   
    ax.set_xlabel('iteration')
    ax.legend()
    
    plt.savefig('img/loss-accuracy.png', bbox_layout='tight')


def plot_decision_bountary(model, X_, y_):
    def make_contourf():
        N = 1000

        x = torch.linspace(0, 1, N)
        y = torch.linspace(0, 1, N)
        x, y = torch.meshgrid(x, y)

        input = torch.stack((x.flatten(), y.flatten()))
        input = input.T

        z = model.forward(input)
        z = z[:,0]
        z = (z > 0.5).float()
        z = z.view(N, N)
    
        return x, y, z

    fig, ax = plt.subplots()

    x, y, z = make_contourf()
    ax.contourf(x, y, z)

    x, y, z = X_[:,0], X_[:,1], y_[:,0]
    ax.scatter(x, y, c=z, s=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.savefig('img/decision-boundary.png', bbox_layout='tight')
