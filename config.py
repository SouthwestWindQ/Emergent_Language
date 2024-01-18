import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=5000)
    parser.add_argument("--minimal_size", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--target_update", type=int, default=5)
    parser.add_argument("--num_episode", type=int, default=1000)
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--state_range", type=int, default=3)
    parser.add_argument("--rule_path", type=str, default="./env")

    return parser.parse_args()
