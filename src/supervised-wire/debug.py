import os
import json
import torch
import torch.nn.functional as F

from tqdm import tqdm

from inside_agent import InsideAgentForInitState, InsideAgentForAction
from outside_agent import OutsideStateModel, OutsideComModel


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

    import pdb; pdb.set_trace()
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


# Read rules which are randomly generated.
with open('rule.json', 'r') as file:
    raw_rule = json.load(file)
    
vocab_size = 128
latent = 256

checkpoint_paths = [os.path.join(f'./checkpoints/v{vocab_size}_l{latent}', name) for name in os.listdir(f'./checkpoints/v{vocab_size}_l{latent}/')]
checkpoint_paths = sorted(checkpoint_paths, key=lambda name: (
    int(name.split('-')[2]), int(name.split('-')[-1].split('.')[0]), name.split('-')[0]
))

# import pdb; pdb.set_trace()
checkpoint_paths = list(filter(lambda name: '32483' in name, checkpoint_paths))

for idx, decoder1_path in tqdm(list(enumerate(checkpoint_paths[::4]))):
    decoder2_path = checkpoint_paths[4*idx+1]
    encoder1_path = checkpoint_paths[4*idx+2]
    encoder2_path = checkpoint_paths[4*idx+3]
    
    encoder1 = torch.load(encoder1_path)
    encoder2 = torch.load(encoder2_path)
    decoder1 = torch.load(decoder1_path)
    decoder2 = torch.load(decoder2_path)
    encoder1.eval()
    encoder2.eval()
    decoder1.eval()
    decoder2.eval()
    
    
    count = 0
    
    symbol1_list = []
    action1_list = []
    symbol2_list = []
    action2_list = []
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                    goal_state = torch.tensor([int(c) for c in raw_rule[f"{i}{j}{k}"]]).unsqueeze(0).cuda()
                    import pdb; pdb.set_trace()
                    symbol1 = F.one_hot(torch.argmax(encoder1(torch.tensor([[i,j,k],]).cuda())), num_classes=vocab_size).float().unsqueeze(0)
                    symbol = torch.argmax(symbol1)
                    symbol1_list.append(symbol.item())
                    
                    action1 = F.one_hot(torch.argmax(decoder1(symbol1, goal_state), dim=-1), num_classes=3).float()
                    action = torch.argmax(action1, dim=1)
                    action1_list.append(action[0].cpu())
                    
                    symbol2 = F.one_hot(torch.argmax(encoder2(action1)), num_classes=vocab_size).float().unsqueeze(0)
                    symbol = torch.argmax(symbol2)
                    symbol2_list.append(symbol.item())
                    
                    action2 = decoder2(symbol2)
                    preds = action2.cpu().argmax(-1)[0]
                    action2_list.append(preds)
                    
                    if ((preds[0].item() + i) % 3 == goal_state[0][0].item() 
                        and (preds[1].item() + j) % 3 == goal_state[0][1].item() 
                        and (preds[2].item() + k) % 3 == goal_state[0][2].item() 
                    ):
                        count += 1

    
with open(f'debug/v{vocab_size}l{latent}lr1e-4.txt', 'a') as file:
    file.write(f"{decoder1_path.split('/')[-1].split('1e-4-')[-1].split('.')[0]} acc: {count}/27\n")
