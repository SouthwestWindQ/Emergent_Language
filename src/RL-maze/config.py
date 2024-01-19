import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=20000)
    parser.add_argument("--minimal_size", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--target_update", type=int, default=20)
    parser.add_argument("--num_episode", type=int, default=2000)
    parser.add_argument("--state_dim", type=int, default=9)
    parser.add_argument("--state_range", type=int, default=3)
    parser.add_argument("--rule_path", type=str, default="./env")
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--env_state_dim", type=int, default=7)
    parser.add_argument("--env_action_dim", type=int, default=4)
    parser.add_argument("--env_state_range", type=int, default=2)
    parser.add_argument("--env_action_range", type=int, default=4)
    return parser.parse_args()
