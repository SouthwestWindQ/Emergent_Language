import os
import torch
import torch.nn.functional as F

from tqdm import tqdm

from inside_agent import InsideAgentForInitState
from outside_agent import OutsideStateModel


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


checkpoint_paths = list(filter(lambda name: 'checkpoint' in name, os.listdir('./')))
checkpoint_paths = sorted(checkpoint_paths, key=lambda name: (
    int(name.split('-')[2]), int(name.split('-')[-1].split('.')[0]), name.split('-')[0]
))

for idx, decoder_path in tqdm(list(enumerate(checkpoint_paths[::2]))):
    encoder_path = checkpoint_paths[2*idx+1]
    
    encoder = torch.load(encoder_path)
    decoder = torch.load(decoder_path)
    encoder.eval()
    decoder.eval()
    
    count = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                    logits = decoder(gumbel_softmax_sample(encoder(torch.tensor([[i,j,k],]).cuda())))
                    preds = logits.cpu().argmax(-1)[0]
                    if preds[0] == i and preds[1] == j and preds[2] == k:
                        count += 1
    
    with open('log.txt', 'a') as file:
        file.write(f"{'-'.join(encoder_path.split('-')[1:]).split('.')[0]} acc: {count/27:.4f}\n")
