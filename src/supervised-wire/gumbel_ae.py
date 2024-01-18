import json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam

from config import parse_args
from utils import gumbel_softmax
from inside_agent import InsideAgentForInitState, InsideAgentForAction
from outside_agent import OutsideStateModel, OutsideComModel


# Read rules which are randomly generated.
with open('rule.json', 'r') as file:
    raw_rule = json.load(file)
    
def rule(init_states, args):
    goal_states = np.zeros((args.batch_size, args.state_dim), dtype=np.int64)
    for i, init_state in enumerate(init_states):
        str_init_state = ''.join(tuple(map(str, init_state)))
        str_goal_state = raw_rule[str_init_state]
        goal_states[i] = np.array([int(c) for c in str_goal_state], dtype=np.int64)
    return goal_states


if __name__ == "__main__":
    args = parse_args()
    
    # Set seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Define the models we will use in our training process. 
    # The parameters transferred for the initialization of each model should be saved in the argument parser later.
    step1_encoder = InsideAgentForInitState(args.state_dim, args.state_range, args.vocab_size, args.latent).cuda()
    step1_decoder = OutsideStateModel(args.state_dim, args.latent, args.state_range, args.vocab_size).cuda()
    step2_encoder = OutsideComModel(args.state_dim, args.state_range, args.latent, args.vocab_size).cuda()
    step2_decoder = InsideAgentForAction(args.state_dim, args.state_range, args.vocab_size, args.latent).cuda()
    
    # Define loss function, optimizer, and progress bar.
    def trainable_parameters():
        # A generator yielding all trainable parameters. Will be used in the optimizer.
        for parameter in step1_encoder.parameters():
            yield parameter
        for parameter in step1_decoder.parameters():
            yield parameter
        for parameter in step2_encoder.parameters():
            yield parameter
        for parameter in step2_decoder.parameters():
            yield parameter
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(trainable_parameters(), lr=args.lr)  # The learning rate should also be saved in the argument parser later!
    progress_bar = tqdm(range(args.episode_num)) 
    
    # Set each model in train mode.
    step1_encoder.train()
    step1_decoder.train()
    step2_encoder.train()
    step2_decoder.train()
    
    for i in progress_bar:
        # Get the initial states and their corresponding goal states.
        inputs = np.random.randint(0, args.state_range, (args.batch_size, args.state_dim))
        goals = rule(inputs, args)
        
        inputs = torch.from_numpy(inputs).cuda()
        goals = torch.from_numpy(goals).cuda()
        
        # Computational steps to get the final actions that the inside agent chooses to take.
        # The final results are the predicted log-probability of each action, which will be used to compute Cross-Entropy loss.
        step1_symbol_distrib = step1_encoder(inputs)
        step1_symbol_onehot = gumbel_softmax(step1_symbol_distrib)
        action_distrib = step1_decoder(step1_symbol_onehot, goals)
        action_onehot = gumbel_softmax(action_distrib)
        step2_symbol_distrib = step2_encoder(action_onehot)
        step2_symbol_onehot = gumbel_softmax(step2_symbol_distrib)
        preds = step2_decoder(step2_symbol_onehot)
        
        # Compute the cross entropy between predictions and the ground truth actions that should be take,
        # and conduct gradient descent through Adam optimizer based on the Cross-Entropy loss.
        gt_actions = (goals - inputs + args.state_range) % args.state_range
        loss = criterion(preds.reshape(-1, args.state_dim), gt_actions.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Set the description of the progress bar, which monitors the loss and the accuracy in the batch.
        progress_bar.set_description(
            'Train | episode {:6d} | loss: {:.3f} | acc: {:.3f}'.format(
                i, loss.item(), torch.sum(torch.all(gt_actions==preds.argmax(-1), dim=1))/args.batch_size
            )
        )
        
        # Each 100 iterations record the loss and accuracy information in the log file.
        if i % 100 == 0:
            with open(f'log/v{args.vocab_size}_l{args.latent}/lr1e-4.txt', 'a') as file:
                file.write('Train | episode {:6d} | loss: {:.3f} | acc: {:.3f}\n'.format(
                    i, loss.item(), torch.sum(torch.all(gt_actions==preds.argmax(-1), dim=1))/args.batch_size
                ))
        
        # Each iteration see whether the model can predict the actions completely rightly in the batch.
        # If yes, store all the checkpoints for further inference.
        # The threshold accuracy and save steps are all hyper-parameters, which I will save in the argument parser later.
        if torch.sum(torch.all(gt_actions==preds.argmax(-1), dim=1)) == args.batch_size:
            torch.save(step1_encoder, f'./checkpoints/v{args.vocab_size}_l{args.latent}/encoder1-lr1e-4-checkpoint-{i}.pth')
            torch.save(step1_decoder, f'./checkpoints/v{args.vocab_size}_l{args.latent}/decoder1-lr1e-4-checkpoint-{i}.pth')
            torch.save(step2_encoder, f'./checkpoints/v{args.vocab_size}_l{args.latent}/encoder2-lr1e-4-checkpoint-{i}.pth')
            torch.save(step2_decoder, f'./checkpoints/v{args.vocab_size}_l{args.latent}/decoder2-lr1e-4-checkpoint-{i}.pth')
