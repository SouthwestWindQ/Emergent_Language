import os
import json
import torch
import itertools
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from config import parse_args
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


if __name__ == "__main__":
    # Read rules which are randomly generated.
    with open('rule.json', 'r') as file:
        raw_rule = json.load(file)
        
    args = parse_args()

    checkpoint_paths = [os.path.join(args.saving_path, f'v{args.vocab_size}_l{args.latent}', name) for name in os.listdir(os.path.join(args.saving_path, f'v{args.vocab_size}_l{args.latent}'))]
    checkpoint_paths = sorted(checkpoint_paths, key=lambda name: (
        int(name.split('-')[-1].split('.')[0]), name.split('-')[-3].split('/')[-1]
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
        all_states = list(itertools.product(np.arange(args.state_range), repeat=args.state_dim))
        
        for state in all_states:
            goal_state = torch.tensor([int(c) for c in raw_rule["".join([str(d) for d in state])]]).unsqueeze(0).cuda()
            symbol1 = F.one_hot(torch.argmax(encoder1(torch.tensor(state).unsqueeze(0).cuda())), num_classes=args.vocab_size).float().unsqueeze(0)
            symbol = torch.argmax(symbol1)
            
            action1 = F.one_hot(torch.argmax(decoder1(symbol1, goal_state), dim=-1), num_classes=args.state_range).float()
            action = torch.argmax(action1, dim=1)
            
            symbol2 = F.one_hot(torch.argmax(encoder2(action1)), num_classes=args.vocab_size).float().unsqueeze(0)
            symbol = torch.argmax(symbol2)
            
            action2 = decoder2(symbol2)
            preds = action2.cpu().argmax(-1)[0]
            
            count += 1
            for idx in range(args.state_dim):
                if (preds[idx].item() + state[idx]) % args.state_range != goal_state[0][idx].item():
                    count -= 1
                    break    
        
        with open(os.path.join(args.logging_path, f'v{args.vocab_size}l{args.latent}', "test.txt"), 'a') as file:
            file.write(f"{decoder1_path} acc: {count}/{len(all_states)}\n")
