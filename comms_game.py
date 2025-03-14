import numpy as np
import random
import torch
from tqdm import tqdm
import os, pickle
import matplotlib.pyplot as plt

from game_env import CommEnv
from model import SpeakerListenerModel2
from ppo import PPOEntireCommunicationFirst

GRID_DIM = 3

MAX_SEQ_LEN = 10  # Maximum length of communication sequence
OBS_SIZE = GRID_DIM * GRID_DIM     # Number of possible observations
VOCAB_SIZE = 3    # Size of "language" for the speaker
MAX_TURNS = 8
EPISODES = 10000
MINI_EPISODES = 4
LR = 1e-5
GAMMA = 0.99
LAMBDA = 0.9
EPS_CLIP = 0.2
ENTROPY_COEF = 0.08
   

def train(env, agent, n_episodes=EPISODES, n_mini_episodes=MINI_EPISODES, start_point=0):
    pbar = tqdm(range(n_episodes), initial=start_point)
    reward_history = []
    loss_history = []

    for ep in pbar:
        # Memory collects data for all mini-episodes in this training episode.
        memory = {
            'obs_tokens': [],         # Constant speaker observations (one per round)
            'comm_tokens': [],        # The full communication string (constant per round)
            'speaker_actions': [],    # List of speaker tokens (the same for every step in the mini-episode)
            'speaker_logprobs': [],
            'speaker_values': [],
            'listener_actions': [],   # Listener’s action per step
            'listener_logprobs': [],
            'listener_values': [],
            'rewards': [],
            'speaker_masks': [],
            'listener_masks': []
        }
        

        for _ in range(n_mini_episodes):
            ep_reward = 0
            spkrobs, obs = env.reset()
            # Convert the observation to a tensor.
            obs_tensor = torch.tensor(obs, dtype=torch.long)
            # Speaker's observation is fixed throughout the rollout.
            speaker_obs_tokens = (obs_tensor + VOCAB_SIZE).unsqueeze(0)  # shape [1, seq_len]
            # Generate the full communication string once using the constant speaker observation.
            (speaker_actions, speaker_logprobs, speaker_values, generated_comm) = agent.create_comms(speaker_obs_tokens)

            done = False
            # In each step, the listener uses its current observation together with the fixed communication.
            while not done:
                # The listener observation is updated at each step.
                listener_obs = torch.tensor(spkrobs, dtype=torch.long).unsqueeze(0)

                (lis_a, lis_lp, lis_val) = agent.select_action_based_comms(
                    generated_comm, encoding_offset=GRID_DIM*GRID_DIM, listener_obs=listener_obs, masked_action=env.invalid_actions()
                )
                action = lis_a.item()
                spkrobs, next_obs, reward, done = env.step(action)

                ep_reward += reward

                # Record the constant speaker data alongside the current listener outputs.
                memory['obs_tokens'].append(speaker_obs_tokens.squeeze(0).tolist())
                memory['comm_tokens'].append(torch.cat([listener_obs, generated_comm + GRID_DIM*GRID_DIM], dim=1).squeeze(0).tolist())
                memory['speaker_actions'].append([a.item() for a in speaker_actions])
                memory['speaker_logprobs'].append([lp.item() for lp in speaker_logprobs])
                memory['speaker_values'].append([val.item() for val in speaker_values])
                memory['listener_actions'].append(lis_a.item())
                memory['listener_logprobs'].append(lis_lp.item())
                memory['listener_values'].append(lis_val.item())
                memory['speaker_masks'].append(1)
                memory['listener_masks'].append(1)
                memory['rewards'].append(reward)

                # Update the listener observation from the environment.
                # (The speaker observation remains constant throughout the rollout.)
                spkrobs = spkrobs  # spkrobs is updated by env.step() and used next iteration

        # Once the episode is over, update the policy.
        # train speaker for 500 episodes, then listener for 500 episodes, then both for 500 episodes, and so on
        
        ldesc = ""
        if ep // 500 % 3 == 0:
            loss = agent.train(memory, train_speaker=True, train_listener=False)
            ldesc = "Speaker"
        elif ep // 500 % 3 == 1:
            loss = agent.train(memory, train_listener=True, train_speaker=False)
            ldesc = "Listener"
        else:
            loss = agent.train(memory, train_speaker=True, train_listener=True)
            ldesc = "Both"


        loss = agent.train(memory)
        loss_history.append(loss)
        reward_history.append(ep_reward)

        pbar.set_description(f"Ep {ep + start_point} Reward: {ep_reward:.2f} Training: {ldesc}")
        if (ep+start_point) % 500 == 0:
            agent.save_to(f"checkpoint{ep+start_point}.pth")
            with open("rewards.pkl", "wb") as f:
                pickle.dump(reward_history, f)

        # Optional testing and checkpointing.
        if ep % 50 == 0:
            print("------")
            test(env, agent, n_episodes=50)
            test(env, PPOEntireCommunicationFirst(SpeakerListenerModel2, obs_size=3, vocab_size=VOCAB_SIZE, n_comm_string=3,  action_size=len(env.action_space), listener_obs_size=GRID_DIM*GRID_DIM, lr=LR, gamma=GAMMA, lmbda=LAMBDA, eps_clip=EPS_CLIP, entropy_coef=ENTROPY_COEF), n_episodes=50)
            render_game(env, agent, render=True)


    return reward_history, loss_history
import time
def render_game(env, agent, render=False):
    done = False
    spkrobs, obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.long)

    # Speaker's observation remains constant throughout the rollout.
    speaker_obs_tokens = (obs + VOCAB_SIZE).unsqueeze(0)

    # Generate the full communication string once.
    (spk_a_list, spk_lp_list, spk_val_list, full_comm) = agent.create_comms(speaker_obs_tokens)

    # For visualization, map tokens to characters.
    comm_map = "abcdefghijklmnopqrstuvwxyz"
    comm_tokens_list = full_comm.squeeze(0).tolist()
    comm_string = "".join([comm_map[token] if 0 <= token < len(comm_map) else "?" 
                           for token in comm_tokens_list])
    
    total_reward = 0

    if render:
        env._init_pygame()
        env.render_mode = "human"
        print("Comm:", comm_string)

    # During the rollout, the speaker's communication remains fixed.
    while not done:
        # Use the current listener observation (from spkrobs) together with the fixed communication.
        listener_obs = torch.tensor(spkrobs, dtype=torch.long).unsqueeze(0)
        (lis_a, lis_lp, lis_val) = agent.select_action_based_comms(
            full_comm, encoding_offset=GRID_DIM*GRID_DIM, listener_obs=listener_obs, masked_action=env.invalid_actions()
        )
        action = lis_a.item()
        spkrobs, next_obs, reward, done = env.step(action)
        total_reward += reward

        if render:
            env.render()
            print("Action:", action, "Reward:", reward, "Steps:", env.steps)
            time.sleep(1)

    if render:
        env.render_mode = None
        env._quit_pygame()

    return total_reward, env.steps, env.count_survivors()

def test(env, agent, n_episodes=10):
    survivors_gotten = 0
    average_steps = 0
    average_reward = 0
    for ep in tqdm(range(n_episodes), desc="Testing"):
        a, b ,c = render_game(env, agent)
        survivors_gotten += (1 - c)
        average_steps += b
        average_reward += a

    print(f"Average Survivors Gotten: {survivors_gotten/n_episodes:.2f}")
    print(f"Average Steps: {average_steps/n_episodes:.2f}")
    print(f"Average Reward: {average_reward/n_episodes:.2f}")

    return average_reward


def graph_checkpoint_accuracy(env, agent, n_episodes=200):
    x = []
    y = []
    for i in range(0, 10000, 500):
        if os.path.exists(f"checkpoint{i}.pth"):
            agent.load_from(f"checkpoint{i}.pth")
            x.append(i)
            y.append(test(env, agent, n_episodes=n_episodes))

    plt.plot(x, y)
    plt.title("Checkpoint vs Accuracy")
    plt.show()

def get_translations(env, agent, n_episodes=10):
    translations = {}

    locations = [(0,1),(0,0)]

    for _ in range(n_episodes):
        for locx in range(GRID_DIM):
            for locy in range(GRID_DIM):

                if locx == 0 and locy == 0:
                    continue

                locations[0] = (locx, locy)
                
                done = False
                spkrobs, obs = env.reset(locations)
                obs = torch.tensor(obs, dtype=torch.long)

                # Speaker's observation remains constant throughout the rollout.
                speaker_obs_tokens = (obs + VOCAB_SIZE).unsqueeze(0)

                # Generate the full communication string once.
                (spk_a_list, spk_lp_list, spk_val_list, full_comm) = agent.create_comms(speaker_obs_tokens)

                # For visualization, map tokens to characters.
                comm_map = "abcdefghijklmnopqrstuvwxyz"
                comm_tokens_list = full_comm.squeeze(0).tolist()
                comm_string = "".join([comm_map[token] if 0 <= token < len(comm_map) else "?" 
                                    for token in comm_tokens_list])
                if not (locx, locy) in translations:
                    translations[(locx, locy)] = []
                translations[(locx, locy)].append(comm_string)

    translation_highest_freq = {}
    tl3 = {}
    for k, v in translations.items():
        hfreq = sorted(list(set(v)), key=v.count, reverse=True)
        translation_highest_freq[k] = [(hfreq[i], v.count(hfreq[i])) for i in range(3)]
        tl3[k] = [(tl, v.count(tl)) for tl in list(set(v))]

    return tl3, translation_highest_freq

if __name__ == "__main__":
    env = CommEnv(grid_size=GRID_DIM, max_timesteps=MAX_TURNS)

    agent = PPOEntireCommunicationFirst(SpeakerListenerModel2, obs_size=3, vocab_size=VOCAB_SIZE, n_comm_string=3,  action_size=len(env.action_space), listener_obs_size=GRID_DIM*GRID_DIM, lr=LR, gamma=GAMMA, lmbda=LAMBDA, eps_clip=EPS_CLIP, entropy_coef=ENTROPY_COEF)

    # graph_checkpoint_accuracy(env, agent, n_episodes=500)


    for i in range(EPISODES, -1000, -500):
        if os.path.exists(f"checkpoint{i}.pth"):
            agent.load_from(f"checkpoint{i}.pth")
            print(f"Loaded from checkpoint{i}.pth")
            break

    # agent.load_from(f"speaker_listener.pth")


    # trans, tl2 = get_translations(env, agent, n_episodes=150)
    # import pprint
    # pprint.pprint(trans)
    # print()
    # pprint.pprint(tl2)

    # agent.load_from(f"speaker_listener.pth")

    if not os.path.exists("speaker_listener.pth"):
        rewards, losses = train(env, agent, n_episodes=10000, start_point=i)
        agent.save_to("speaker_listener.pth")

        with open("losses.pkl", "wb") as f:
            pickle.dump(losses, f)
        plt.plot(losses)
        plt.title("Losses")
        plt.show()
        # save losses
        with open("rewards.pkl", "wb") as f:
            pickle.dump(rewards, f)
        plt.plot(rewards)
        plt.title("Rewards")
        plt.show()

    
    with open("rewards.pkl", "rb") as f:
        rewards = pickle.load(f)

    # perform running average with windoe of 50
    window = 100
    rewards = [sum(rewards[i:i+window])/window for i in range(len (rewards)-window)]

    plt.plot(rewards)
    plt.title("Rewards")
    plt.show()

    test(env, agent, n_episodes=500)
    test(env, PPOEntireCommunicationFirst(SpeakerListenerModel2, obs_size=3, vocab_size=VOCAB_SIZE, n_comm_string=3,  action_size=len(env.action_space), listener_obs_size=GRID_DIM*GRID_DIM, lr=LR, gamma=GAMMA, lmbda=LAMBDA, eps_clip=EPS_CLIP, entropy_coef=ENTROPY_COEF), n_episodes=500)

    # for _ in range(5): render_game(env, agent, render=True)

    # agent.load_from(f"checkpoint1000.pth")
    # test(env, agent, n_episodes=75)
    # ts, _ = get_translations(env, agent, n_episodes=10)
    # print(ts)