import torch
from matplotlib import pyplot as plt


def plot_loss_accuracy(loss, accuracy):
    ''' Plots loss vs. accuracy curves '''

    fig, ax1 = plt.subplots()

    color = 'tab:purple'
    x, y = zip(*loss)
    ax1.plot(x, y, color=color)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss (train)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
   
    ax2 = ax1.twinx()

    color = 'tab:olive'
    x,  y = zip(*accuracy)
    ax2.plot(x, y, '-x', color=color)
    ax2.set_ylabel('accuracy (test)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.savefig('img/loss-accuracy.png', bbox_layout='tight')


def plot_decision_bountary(model, X_, y_):
    '''' Plots decision boundary '''
    
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
