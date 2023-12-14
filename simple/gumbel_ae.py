import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam

from inside_agent import InsideAgentForInitState
from outside_agent import OutsideStateModel


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


if __name__ == "__main__":
    
    batch_size = 16
    episode_num = 100000
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    encoder = InsideAgentForInitState(3, 3, 64, 128).cuda()
    decoder = OutsideStateModel(3, 128, 3, 64).cuda()
    
    def trainable_parameters():
        for parameter in encoder.parameters():
            yield parameter
        for parameter in decoder.parameters():
            yield parameter
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(trainable_parameters(), lr=1e-4)
    progress_bar = tqdm(range(episode_num)) 
    encoder.train()
    decoder.train()
    
    for i in progress_bar:
        inputs = np.random.randint(0, 3, (batch_size, 3))
        inputs = torch.from_numpy(inputs).cuda()
        symbol_distrib = encoder(inputs)
        symbol_onehot = gumbel_softmax(symbol_distrib)
        preds = decoder(symbol_onehot)
        loss = criterion(preds.reshape(-1,3), inputs.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_description(
            'Train | episode {:6d} | loss: {:.3f} | acc: {:.3f}'.format(
                i, loss.item(), torch.sum(torch.all(inputs==preds.argmax(-1), dim=1))/batch_size
            )
        )
        if i % 100 == 0:
            with open('log/lr1e-4.txt', 'a') as file:
                file.write('Train | episode {:6d} | loss: {:.3f} | acc: {:.3f}\n'.format(
                    i, loss.item(), torch.sum(torch.all(inputs==preds.argmax(-1), dim=1))/batch_size
                ))
        if i % 10 == 0:
            if torch.sum(torch.all(inputs==preds.argmax(-1), dim=1)) == batch_size:
                torch.save(encoder, f'encoder-lr1e-4-checkpoint-{i}.pth')
                torch.save(decoder, f'decoder-lr1e-4-checkpoint-{i}.pth')
