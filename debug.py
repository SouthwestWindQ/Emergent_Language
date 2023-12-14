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

checkpoint_paths = [os.path.join('./checkpoints', name) for name in os.listdir('./checkpoints/')]
checkpoint_paths = sorted(checkpoint_paths, key=lambda name: (
    int(name.split('-')[2]), int(name.split('-')[-1].split('.')[0]), name.split('-')[0]
))

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
    for i in range(3):
        for j in range(3):
            for k in range(3):
                    goal_state = torch.tensor([int(c) for c in raw_rule[f"{i}{j}{k}"]]).unsqueeze(0).cuda()
                    logits = gumbel_softmax(encoder1(torch.tensor([[i,j,k],]).cuda()))
                    logits = gumbel_softmax(decoder1(logits, goal_state))
                    logits = gumbel_softmax(encoder2(logits))
                    logits = decoder2(logits)
                    preds = logits.cpu().argmax(-1)[0]
                    if ((preds[0].item() + i) % 3 == goal_state[0][0].item() 
                        and (preds[1].item() + j) % 3 == goal_state[0][1].item() 
                        and (preds[2].item() + k) % 3 == goal_state[0][2].item() 
                    ):
                        count += 1
    
                    with open('debug.txt', 'a') as file:
                        file.write(f"initial: {i} {j} {k} preds: {preds[0]} {preds[1]} {preds[2]} goal {goal_state[0][0]} {goal_state[0][1]} {goal_state[0][2]}\n{logits}\n\n")
