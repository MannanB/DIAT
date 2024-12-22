import gym
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# Define PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Define TransformerDecoder
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_hidden_layer, dropout):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_hidden_layer, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x

# Define ActorCriticListener
class ActorCriticListener(nn.Module):
    def __init__(self, vocab_size, actions):
        super(ActorCriticListener, self).__init__()
        self.d_model = 64
        self.heads = 4
        self.transformer = TransformerDecoder(vocab_size, self.d_model, self.heads, 128, 0.1)
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.transformer(x)
        x = x.squeeze(1).mean(dim=0)
        x = self.fc(x)
        logits = self.actor(x).squeeze()
        value = self.critic(x)
        return logits, value

# Define ActorCriticSpeaker
class ActorCriticSpeaker(nn.Module):
    def __init__(self, input_size, vocab_size):
        super(ActorCriticSpeaker, self).__init__()
        self.d_model = 64
        self.heads = 4
        self.transformer = TransformerDecoder(vocab_size + input_size, self.d_model, self.heads, 128, 0.1)
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, vocab_size)
        self.critic = nn.Linear(128, 1)


    def forward(self, nx):
        x = self.transformer(nx)
        x = x.squeeze(1).mean(dim=0)
        x = self.fc(x)
        logits = self.actor(x).squeeze()
        value = self.critic(x)
        print(nx)
        return F.one_hot(nx[0].item(), 2), value

# Define PPO
class PPO:
    def __init__(self, nn, gamma=0.99, eps_clip=0.1, lr=1e-4, lmbda=0.95, entropy_coef=0.05):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef
        self.policy = nn()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.old_policy = nn()
        self.old_policy.load_state_dict(self.policy.state_dict())

    def convert_obs(self, obs):
        if isinstance(obs, list):
            return torch.stack(obs)
        return obs

    def select_action(self, obs):
        logits, _ = self.old_policy(self.convert_obs(obs))
        # logits = logits - logits.max(dim=-1, keepdim=True)[0]  # Stabilize logits
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
        # print(probs)
        if torch.isnan(probs).any():
            print("NaN detected in probs. Logits:", logits)
            raise ValueError("NaN detected in probabilities")
        
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        return actions, dist.log_prob(actions)
    
    def select_action_deterministic(self, obs):
        logits, _ = self.old_policy(self.convert_obs(obs))
        # logits = logits - logits.max(dim=-1, keepdim=True)[0]  # Stabilize logits
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
        # print(probs)
        if torch.isnan(probs).any():
            print("NaN detected in probs. Logits:", logits)
            raise ValueError("NaN detected in probabilities")
        
        actions = probs.argmax()
        log_prob = torch.log(probs.flatten()[actions])
        return actions, log_prob

    def compute_gae(self, rewards, masks, values, next_value):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            advantages.insert(0, gae)
            next_value = values[i]
        return torch.FloatTensor(advantages), torch.FloatTensor(advantages) + values

    def train(self, memory, optimize=True):
        obs, actions, log_probs, rewards, masks, values = memory

        # Normalize rewards
        rewards = np.array(rewards)
        if rewards.std() > 1e-8:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards = rewards - rewards.mean()

        # Compute advantages and returns
        next_value = self.policy(self.convert_obs(obs[-1:]))[1].item()
        log_probs = torch.FloatTensor(log_probs).detach()
        values = torch.FloatTensor(values).detach()

        advantages, returns = self.compute_gae(rewards, masks, values, next_value)

        # Normalize advantages
        # if advantages.std() > 1e-8:
        #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # else:
        #     advantages = advantages - advantages.mean()

        for _ in range(4):  # Multiple updates per batch
            logits, state_values = self.policy(self.convert_obs(obs))
            # logits = logits - logits.max(dim=-1, keepdim=True)[0]  # Stabilize logits
            probs = torch.softmax(logits, dim=-1)
            probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)  # Clamp probabilities
            
            # Debug for NaN issues
            if torch.isnan(probs).any():
                print("NaN detected in probs. Logits:", logits)
                raise ValueError("NaN detected in probabilities")
            
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(torch.LongTensor(actions))
            entropy = dist.entropy().mean()

            # Compute ratio
            ratio = torch.exp(new_log_probs - log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Clipped value loss
            value_pred_clipped = values + (returns - values).clamp(-self.eps_clip, self.eps_clip)
            value_loss_clipped = (value_pred_clipped - returns).pow(2)
            critic_loss = 0.5 * torch.max(value_loss_clipped, (state_values - returns).pow(2)).mean()

            # Total loss
            loss = actor_loss + critic_loss - self.entropy_coef * entropy

            # Optimize policy
            if optimize:
                self.optimizer.zero_grad()
                loss.backward()
                # for param in self.policy.parameters():
                #     if param.grad is not None:
                #         param.grad.data.clamp_(-0.5, 0.5)  # Gradient clipping
                self.optimizer.step()

        # Update old policy
        if optimize:
            self.old_policy.load_state_dict(self.policy.state_dict())
        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()

    def save_to(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load_from(self, path):
        self.policy.load_state_dict(torch.load(path))

from collections import defaultdict
# Define CommEnv
class CommEnv:
    def __init__(self, obs_size, ts_limit=10):
        self.ts_limit = ts_limit
        self.obs_size = obs_size
        self.current_obs = None
        self.current_comm = []
        self.reset()

    def reset(self, obj=None):
        if obj is None:
            obj = random.randint(0, self.obs_size - 1)
        self.current_obs = torch.tensor([obj], dtype=torch.int)
        self.current_comm = []
        self.timesteps = 0
        return self.current_obs

    def step(self, actions):
        self.timesteps += 1
        done = self.timesteps >= self.ts_limit
        speaker_action = actions["speaker"]
        listener_action = actions["listener"]

        self.current_comm.append(speaker_action)

        if listener_action == self.current_obs.argmax().item():
            reward = 1
            done = True
        else:
            reward = -1#self.timesteps / (self.ts_limit * .9)
            done = False or done
        return self.current_obs, reward, done, {}

# Training and Testing
obs_size = 2
language_size = 2
speaker = lambda: ActorCriticSpeaker(obs_size , language_size)
listener = lambda: ActorCriticListener(language_size, obs_size)
ppo_speaker = PPO(speaker)
ppo_listener = PPO(listener)

def train(env, n_episodes=1000, mini_epochs=4):
    pbar = tqdm(range(n_episodes))
    loss_listeners = []
    loss_speakers = []
    average_rewards = []

    learning_speaker = True

    env.ts_limit = 1

    for episode in pbar:
        obs = env.reset()
        done = False
        memory_speaker = ([], [], [], [], [], [])
        memory_listener = ([], [], [], [], [], [])
        average_reward = 0
        comms = []
        for _ in range(mini_epochs):
            while not done:
                if len(comms) > 0:
                    obs_with_comm = torch.cat([obs, torch.stack(comms)])
                else:
                    obs_with_comm = obs 
                speaker_action, speaker_log_prob = ppo_speaker.select_action_deterministic(obs_with_comm)
                comms.append(speaker_action)
                listener_action, listener_log_prob = ppo_listener.select_action_deterministic(torch.stack(comms))
                listener_action = listener_action.item()
                actions = {"speaker": speaker_action, "listener": listener_action}
                next_obs, reward, done, _ = env.step(actions)
                memory_speaker[0].append(obs)
                memory_speaker[1].append(speaker_action)
                memory_speaker[2].append(speaker_log_prob)
                memory_speaker[3].append(reward)
                memory_speaker[4].append(0 if done else 1)
                memory_speaker[5].append(ppo_speaker.policy(obs_with_comm)[1].item())
                memory_listener[0].append(speaker_action)
                memory_listener[1].append(listener_action)
                memory_listener[2].append(listener_log_prob)
                memory_listener[3].append(reward)
                memory_listener[4].append(0 if done else 1)
                memory_listener[5].append(ppo_listener.policy(torch.stack(comms))[1].item())
                average_reward += reward
                obs = next_obs
        if episode % 100:
            learning_speaker = not learning_speaker
            env.ts_limit += 0

        average_rewards.append(average_reward / mini_epochs)
        
        loss_speaker, actor_loss_speaker, critic_loss_speaker, _ = ppo_speaker.train(memory_speaker, optimize=True)
        loss_listener, actor_loss_listener, critic_loss_listener, _ = ppo_listener.train([
            torch.stack(memory_listener[0]),
            torch.tensor(memory_listener[1]),
            torch.stack(memory_listener[2]),
            memory_listener[3],
            memory_listener[4],
            torch.tensor(memory_listener[5])
        ], optimize=True)
        loss_speakers.append(loss_speaker)
        loss_listeners.append(loss_listener)
        pbar.set_description(f"Episode: {episode + 1}, Loss Speaker: {loss_speaker}, Loss Listener: {loss_listener}")
    return loss_speakers, loss_listeners, average_rewards

def test(env, n_episodes=10):
    comm_map = "abcdefghijklmnopqrstuvwxyz"
    obs_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        comms = []
        comm_string = ""
        while not done:
            speaker_action, _ = ppo_speaker.select_action_deterministic(obs)
            comm_string += comm_map[speaker_action]
            comms.append(speaker_action)
            listener_action, _ = ppo_listener.select_action_deterministic(torch.stack(comms))
            actions = {"speaker": speaker_action, "listener": listener_action}
            obs, reward, done, _ = env.step(actions)
            print(f"Episode: {episode + 1}, Communication: {comm_string}, Target: {obs_map[obs.argmax().item()]}, Prediction: {obs_map[listener_action.item()]}")

def find_translations(env, n_episodes=10):
    comm_map = "abcdefghijklmnopqrstuvwxyz"
    obs_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    translations = {}
    env.ts_limit = 1
    for episode in range(n_episodes):
        for i in range(obs_size):
            obs = env.reset(i)
            done = False
            comms = []
            comm_string = ""
            while not done:
                speaker_action, _ = ppo_speaker.select_action_deterministic(obs)
            
                comm_string += comm_map[speaker_action]
                comms.append(speaker_action)
                print(torch.stack(comms))
                listener_action, _ = ppo_listener.select_action_deterministic(torch.stack(comms))
                print(f"for observation {obs}, speaker action: {speaker_action.item()}, listener action: {listener_action.item()}")
                actions = {"speaker": speaker_action, "listener": listener_action}
                obs, reward, done, _ = env.step(actions)
            if obs_map[i] not in translations:
                translations[obs_map[i]] = []
            if reward > 0:
                translations[obs_map[i]].append(comm_string)
    return translations

env = CommEnv(obs_size)
import os
if not os.path.exists("listener.pth"):
    losses_speaker, losses_listner, average_rewards = train(env, mini_epochs=1)

    import matplotlib.pyplot as plt
    plt.plot(losses_speaker, label="Speaker")
    plt.plot(losses_listner, label="Listener")
    plt.legend()
    plt.title("Speaker Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(average_rewards)
    plt.title("Average Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

    ppo_listener.save_to("listener.pth")
    ppo_speaker.save_to("speaker.pth")
else:
    ppo_listener.load_from("listener.pth")
    ppo_speaker.load_from("speaker.pth")
    for i in range(2):
        for _ in range(10):
            print(ppo_speaker.select_action_deterministic(torch.tensor([i]))[0].item())
        print("f")
    for i in range(2):
        for _ in range(10):
            print(ppo_listener.select_action_deterministic(torch.tensor([i]))[0].item())
        print("f")

    env.ts_limit = 1
    test(env)
    translations = find_translations(env)
    for obs, comms in translations.items():
        print(f"Observation: {obs}, Communications: {comms}")
