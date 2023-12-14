import json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam

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
    
def rule(init_states):
    goal_states = np.zeros((batch_size, 3), dtype=np.int64)
    for i, init_state in enumerate(init_states):
        str_init_state = ''.join(tuple(map(str, init_state)))
        str_goal_state = raw_rule[str_init_state]
        goal_states[i] = np.array([int(c) for c in str_goal_state], dtype=np.int64)
    return goal_states


if __name__ == "__main__":
    batch_size = 16
    episode_num = 200000
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    step1_encoder = InsideAgentForInitState(3, 3, 64, 128).cuda()
    step1_decoder = OutsideStateModel(3, 128, 3, 64).cuda()
    step2_encoder = OutsideComModel(3, 3, 128, 64).cuda()
    step2_decoder = InsideAgentForAction(3, 3, 64, 128).cuda()
    
    def trainable_parameters():
        for parameter in step1_encoder.parameters():
            yield parameter
        for parameter in step1_decoder.parameters():
            yield parameter
        for parameter in step2_encoder.parameters():
            yield parameter
        for parameter in step2_decoder.parameters():
            yield parameter
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(trainable_parameters(), lr=1e-4)
    progress_bar = tqdm(range(episode_num)) 
    
    step1_encoder.train()
    step1_decoder.train()
    step2_encoder.train()
    step2_decoder.train()
    
    for i in progress_bar:
        inputs = np.random.randint(0, 3, (batch_size, 3))
        goals = rule(inputs)
        
        inputs = torch.from_numpy(inputs).cuda()
        goals = torch.from_numpy(goals).cuda()
        
        step1_symbol_distrib = step1_encoder(inputs)
        step1_symbol_onehot = gumbel_softmax(step1_symbol_distrib)
        action_distrib = step1_decoder(step1_symbol_onehot, goals)
        action_onehot = gumbel_softmax(action_distrib)
        step2_symbol_distrib = step2_encoder(action_onehot)
        step2_symbol_onehot = gumbel_softmax(step2_symbol_distrib)
        preds = step2_decoder(step2_symbol_onehot)
        gt_actions = (goals - inputs + 3) % 3 
        
        loss = criterion(preds.reshape(-1,3), gt_actions.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_description(
            'Train | episode {:6d} | loss: {:.3f} | acc: {:.3f}'.format(
                i, loss.item(), torch.sum(torch.all(gt_actions==preds.argmax(-1), dim=1))/batch_size
            )
        )
        if i % 100 == 0:
            with open('log/lr1e-4.txt', 'a') as file:
                file.write('Train | episode {:6d} | loss: {:.3f} | acc: {:.3f}\n'.format(
                    i, loss.item(), torch.sum(torch.all(gt_actions==preds.argmax(-1), dim=1))/batch_size
                ))
        if i % 10 == 0:
            if torch.sum(torch.all(gt_actions==preds.argmax(-1), dim=1)) == batch_size:
                torch.save(step1_encoder, f'./checkpoints/encoder1-lr1e-4-checkpoint-{i}.pth')
                torch.save(step1_decoder, f'./checkpoints/decoder1-lr1e-4-checkpoint-{i}.pth')
                torch.save(step2_encoder, f'./checkpoints/encoder2-lr1e-4-checkpoint-{i}.pth')
                torch.save(step2_decoder, f'./checkpoints/decoder2-lr1e-4-checkpoint-{i}.pth')
