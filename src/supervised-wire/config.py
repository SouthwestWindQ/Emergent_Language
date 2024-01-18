import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed used in this experiment.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size used in this experiment.")
    parser.add_argument("--episode_num", type=int, default=100000, help="The number of iterations of the game.")
    parser.add_argument("--vocab_size", type=int, default=64, help="The size of vocabulary.")
    parser.add_argument("--latent", type=int, default=128, help="The latent shape used in our MLPs.")
    parser.add_argument("--state_dim", type=int, default=3, help="The number of digits in one state.")
    parser.add_argument("--state_range", type=int, default=3, help="The number of possible values in one digit.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate used in the training process.")
    
    return parser.parse_args()
