import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def prediction(input):

    inception_net = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_net.eval()

    upsample = nn.Upsample(size=(299, 299), mode='bilinear').to(device)

    output = upsample(input)
    output = inception_net(output)
    output = F.softmax(output)

    return output.data.cpu().numpy()

def inception_score(generated_images, device, batch_size=64):
    N = generated.size(0)
    dataloader = torch.utils.data.DataLoader(generated_images, batch_size=batch_size)

    pred = np.zeros((N, 1000))

    for i, data in enumerate(dataloader, 0):
        data = data.to(device)
        batch_size_i = data.size(0)
        pred[i*batch_size:i*batch_size + batch_size_i] = get_pred(data)

    p_y = np.mean(pred, axis=0)
    scores = []
    for i in range(pred.shape[0]):
        p_yx = pred[i, :]
        scores.append(entropy(p_yx, p_y))
    return np.exp(np.mean(scores))
