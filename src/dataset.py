import math
import torch


def gen_dataset(N):
    center = torch.tensor([0.5, 0.5])
    radius = torch.tensor(1 / math.sqrt(2 * math.pi))

    X = torch.empty((N, 2)).uniform_()
    y = torch.empty((N, 2)).zero_()
    
    y[:,0] = (X[:,0] - center[0]) ** 2 + (X[:,1] - center[1]) ** 2 < radius ** 2
    y[:,1] = (X[:,0] - center[0]) ** 2 + (X[:,1] - center[1]) ** 2 > radius ** 2

    y = y.float()

    return X, y
