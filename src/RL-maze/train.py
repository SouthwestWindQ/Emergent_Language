import torch
import random
import numpy as np

from tqdm import tqdm

from config import parse_args
from environment import Wire3Env
from gridworld import GridWorld
from agent import ReplayBuffer, DQN
import os
import wandb



import matplotlib.pyplot as plt


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = GridWorld()

    random.seed(random.randint(0,100))
    np.random.seed(random.randint(0,100))
    env.seed(random.randint(0,100))
    torch.manual_seed(random.randint(0,100))

    replay_buffer_incom = ReplayBuffer(args.capacity)
    replay_buffer_inact = ReplayBuffer(args.capacity)
    replay_buffer_out = ReplayBuffer(args.capacity)

    state_dim = args.env_state_dim * args.env_state_dim + 2
    state_range = 2

    action_dim = args.env_action_dim
    action_range = args.env_action_range

    agent = DQN(state_dim=state_dim, hidden_dim=args.hidden_dim, action_dim=action_dim,
                state_range=state_range, action_range=action_range, vocab_size=args.vocab_size,
                lr=args.lr, gamma=args.gamma, epsilon=args.epsilon, target_update=args.target_update,
                device=device)

    return_list = []
    pic_return_list = []
    step_list = []
    x = []
    cnt = 0
    train_cnt = 0
    for i in range(10):
        with tqdm(total=int(args.num_episode / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episode / 10)):
                episode_return = 0
                wall, now_pos, goal_state = env.reset()
                done = False
                cnt += 1
                x.append(cnt)
                last_reward = None
                last_incom_symbol = None
                last_out_symbol = None
                last_done = None
                while not done:
                    mod_state = np.squeeze(wall.reshape(1,-1))
                    mod_goal_state = np.concatenate((mod_state, goal_state))
                    mod_state = np.concatenate((mod_state, now_pos))

                    incom_symbol, out_symbol, action = agent.take_action(mod_state, mod_goal_state)

                    next_pos, reward, done = env.step(action)
                    next_state = np.squeeze(wall.reshape(1,-1))
                    next_state = np.concatenate((next_state, now_pos))

                    # if i_episode == 10:
                    #     print(action, next_state, reward)
                    done = 0 if done == False else 1
                    done = np.array(done, dtype=np.int32)
                    reward = np.array(reward)
                    replay_buffer_incom.add(mod_state, incom_symbol, reward, next_state, mod_goal_state, done)
                    if last_incom_symbol is not None:
                        replay_buffer_out.add(last_incom_symbol, out_symbol, last_reward, incom_symbol, mod_goal_state, last_done)
                    if last_out_symbol is not None:
                        replay_buffer_inact.add(last_out_symbol, action, last_reward, out_symbol, mod_goal_state, last_done)


                    state = next_state
                    episode_return += reward
                    train_cnt += 1

                    last_reward = reward
                    last_done = done
                    last_out_symbol = out_symbol
                    last_incom_symbol = incom_symbol
                    if replay_buffer_incom.size() > args.minimal_size and train_cnt % args.interval == 0:
                        s1, a1, r1, ns1, g1, d1 = replay_buffer_incom.sample(args.batch_size)
                        incom_data = {
                            'state':       s1,
                            'action':      a1,
                            'next_state':  ns1,
                            'reward':      r1,
                            'goal_state':  g1,
                            'done':        d1,
                        }
                        s2, a2, r2, ns2, g2, d2 = replay_buffer_out.sample(args.batch_size)
                        out_data = {
                            'state': s2,
                            'action': a2,
                            'next_state': ns2,
                            'reward': r2,
                            'goal_state': g2,
                            'done': d2,
                        }
                        s3, a3, r3, ns3, g3, d3 = replay_buffer_inact.sample(args.batch_size)
                        inact_data = {
                            'state': s3,
                            'action': a3,
                            'next_state': ns3,
                            'reward': r3,
                            'goal_state': g3,
                            'done': d3,
                        }
                        agent.update(incom_data, out_data, inact_data)
                incom_symbol, out_symbol, action = agent.take_action(state, mod_goal_state)
                if last_incom_symbol is not None:
                    replay_buffer_out.add(last_incom_symbol, out_symbol, last_reward, incom_symbol, mod_goal_state,
                                          last_done)
                if last_out_symbol is not None:
                    replay_buffer_inact.add(last_out_symbol, action, last_reward, out_symbol, mod_goal_state, last_done)

                return_list.append(episode_return)
                pic_return_list.append(np.mean(return_list[-10:]))
                step_list.append(env.now_step)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':  '%d' % (args.num_episode / 10 * i + i_episode + 1),
                        'return':   '%.3f' % np.mean(return_list[-10:]),
                    })
                    # with open('debug/output1.txt', 'a') as f:
                    #     f.write(f'{args.num_episode / 10 * i + i_episode + 1}: {np.mean(return_list[-10:])}')
                    #     f.write('\n')
                pbar.update(1)
            agent.epsilon = agent.epsilon * 0.99



    print(f"mean_step = {np.mean(step_list[-1000:])}")
    print(f"mean_return = {np.mean(return_list[-1000:])}")
    plt.plot(x, step_list)
    plt.show()

    plt.plot(x, pic_return_list)
    plt.show()