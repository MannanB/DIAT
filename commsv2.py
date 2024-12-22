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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_hidden_layer, dropout=dropout, batch_first=True )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        print("-----")
        x = self.embedding(x)
        print("EMBEDDING:", x.shape)
        x = self.pos_encoder(x)
        print("POS ENCODER:", x.shape)
        x = self.transformer_encoder(x)
        print("TRANSFORMER:", x.shape)
        print("-----")
        return x

# Define Unified ActorCritic Model
class UnifiedActorCritic(nn.Module):
    def __init__(self, input_size, vocab_size, obs_size):
        super(UnifiedActorCritic, self).__init__()
        self.d_model = 64
        self.heads = 4

        # Speaker network
        self.speaker_transformer = TransformerDecoder(input_size + vocab_size, self.d_model, self.heads, 128, 0.1)
        self.speaker_fc = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU()
        )
        self.speaker_actor = nn.Linear(128, vocab_size)

        # Listener network
        self.listener_transformer = TransformerDecoder(vocab_size, self.d_model, self.heads, 128, 0.1)
        self.listener_fc = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU()
        )
        self.listener_actor = nn.Linear(128, obs_size)
        self.listener_critic = nn.Linear(128, 1)

    def forward(self, obs, comms):
        if isinstance(comms, list):
            comms = torch.tensor(comms, dtype=torch.long)
        # Speaker forward pass
        if comms.numel() > 0:  # Check if comms has elements
            print("OBS+COMMS", obs, comms)
            obs_with_comm = torch.cat([obs.unsqueeze(0), comms.unsqueeze(0)], dim=-1)
        else:
            obs_with_comm = obs.unsqueeze(0)  # Add a batch dimension for compatibility

        print("Speaker in:", obs_with_comm.shape)
        speaker_out = self.speaker_transformer(obs_with_comm)
        print("Speaker out:", speaker_out.shape)
        speaker_out = speaker_out.squeeze(0).mean(dim=0)
        print("Speaker out Mean:", speaker_out.shape)
        speaker_out = self.speaker_fc(speaker_out)
        speaker_logits = self.speaker_actor(speaker_out)

        new_comms = torch.cat([comms, torch.argmax(speaker_logits).unsqueeze(0)])
        # Listener forward pass
        listener_out = self.listener_transformer(new_comms.unsqueeze(0))
        listener_out = listener_out.squeeze(0).mean(dim=0)
        listener_out = self.listener_fc(listener_out)
        listener_logits = self.listener_actor(listener_out)
        listener_value = self.listener_critic(listener_out)

        return speaker_logits, listener_logits, listener_value


# Define PPO
class PPO:
    def __init__(self, model, gamma=0.99, eps_clip=0.1, lr=1e-4, lmbda=0.95, entropy_coef=0.05):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def select_action(self, obs, comms):
        speaker_logits, listener_logits, _ = self.model(obs, comms)
        print("LOGITS:", speaker_logits, listener_logits)
        speaker_probs = torch.softmax(speaker_logits, dim=-1)
        listener_probs = torch.softmax(listener_logits, dim=-1)
        print("PROBS:", speaker_probs, listener_probs)

        speaker_dist = torch.distributions.Categorical(speaker_probs)
        listener_dist = torch.distributions.Categorical(listener_probs)

        speaker_action = speaker_dist.sample()
        listener_action = listener_dist.sample()

        return speaker_action, speaker_dist.log_prob(speaker_action), listener_action, listener_dist.log_prob(listener_action)

    def compute_gae(self, rewards, masks, values, next_value):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            advantages.insert(0, gae)
            next_value = values[i]
        return torch.FloatTensor(advantages), torch.FloatTensor(advantages) + values

    def train(self, memory):
        obs, comms, speaker_actions, speaker_log_probs, listener_actions, listener_log_probs, rewards, masks, values = memory

        # Ensure obs and comms are not empty
        if not obs or not comms:
            raise ValueError("Observation or comms memory is empty.")

        # Get the last observation and comms
        last_obs = obs[-1]      # Shape: [1, 1]
        last_comms = comms[-1]  # Shape: [1, N]

        # No unsqueeze here; ensure they are [batch_size, seq_length]
        # For example, [1, 1] and [1, N]

        # Get the next value
        _, _, next_value = self.model(last_obs, last_comms)
        next_value = next_value.item()

        # Convert lists to tensors
        obs_tensor = torch.stack(obs)          # Shape: [T, 1, 1]
        comms_tensor = torch.stack(comms)      # Shape: [T, 1, N]
        speaker_actions = torch.stack(speaker_actions)         # Shape: [T]
        speaker_log_probs = torch.stack(speaker_log_probs)     # Shape: [T]
        listener_actions = torch.stack(listener_actions)       # Shape: [T]
        listener_log_probs = torch.stack(listener_log_probs)   # Shape: [T]
        rewards = torch.tensor(rewards, dtype=torch.float)     # Shape: [T]
        masks = torch.tensor(masks, dtype=torch.float)         # Shape: [T]
        values = torch.tensor(values, dtype=torch.float)       # Shape: [T]

        # Compute GAE
        advantages, returns = self.compute_gae(rewards, masks, values, next_value)

        for _ in range(4):  # Multiple updates per batch
            speaker_logits, listener_logits, state_values = self.model(obs_tensor, comms_tensor)

            speaker_probs = torch.softmax(speaker_logits, dim=-1)
            listener_probs = torch.softmax(listener_logits, dim=-1)

            speaker_dist = torch.distributions.Categorical(speaker_probs)
            listener_dist = torch.distributions.Categorical(listener_probs)

            new_speaker_log_probs = speaker_dist.log_prob(speaker_actions)
            new_listener_log_probs = listener_dist.log_prob(listener_actions)

            speaker_ratio = torch.exp(new_speaker_log_probs - speaker_log_probs)
            listener_ratio = torch.exp(new_listener_log_probs - listener_log_probs)

            surr1 = speaker_ratio * advantages
            surr2 = torch.clamp(speaker_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            speaker_loss = -torch.min(surr1, surr2).mean()

            surr1 = listener_ratio * advantages
            surr2 = torch.clamp(listener_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            listener_loss = -torch.min(surr1, surr2).mean()

            value_loss = (state_values.squeeze(-1) - returns).pow(2).mean()

            entropy_loss = speaker_dist.entropy().mean() + listener_dist.entropy().mean()

            loss = speaker_loss + listener_loss + 0.5 * value_loss - self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

# Define CommEnv
class CommEnv:
    def __init__(self, obs_size, ts_limit=10):
        self.ts_limit = ts_limit
        self.obs_size = obs_size
        self.current_obs = None
        self.current_comm = torch.empty((0,), dtype=torch.long)
        self.reset()

    def reset(self, obj=None):
        if obj is None:
            obj = random.randint(0, self.obs_size - 1)
        self.current_obs = torch.tensor([obj], dtype=torch.long)
        self.current_comm = torch.empty((0,), dtype=torch.long)
        self.timesteps = 0
        return self.current_obs

    def step(self, actions):
        self.timesteps += 1
        done = self.timesteps >= self.ts_limit
        speaker_action = actions["speaker"]
        listener_action = actions["listener"]

        # Ensure listener_action is a scalar
        if listener_action.numel() != 1:
            raise ValueError(f"Expected listener_action to have 1 element, but got {listener_action.numel()} elements")

        listener_action_scalar = listener_action.item()

        # Append speaker_action to current_comm
        self.current_comm = torch.cat([self.current_comm, speaker_action.unsqueeze(0)], dim=0)
        print("ACTION+OBS", listener_action_scalar, self.current_obs)

        if listener_action_scalar == self.current_obs.item():
            reward = 1
            done = True
        else:
            reward = -1
            done = False or done

        return self.current_obs, reward, done, {}

# Training loop
obs_size = 2
vocab_size = 2

model = UnifiedActorCritic(obs_size, vocab_size, obs_size)
ppo = PPO(model)
env = CommEnv(obs_size)

def train(ppo, env, n_episodes=1000):
    pbar = tqdm(range(n_episodes))
    for episode in pbar:
        obs = env.reset()
        done = False
        memory = ([], [], [], [], [], [], [], [], [])
        comms = torch.empty((0,), dtype=torch.long)  # Initialize as an empty tensor

        while not done:
            speaker_action, speaker_log_prob, listener_action, listener_log_prob = ppo.select_action(obs, comms)
            print("ACTIONS:", speaker_action, listener_action)
            actions = {"speaker": speaker_action, "listener": listener_action}

            next_obs, reward, done, _ = env.step(actions)
            memory[0].append(obs)
            memory[1].append(comms.clone())  # Store comms as a clone to avoid overwriting
            memory[2].append(speaker_action)
            memory[3].append(speaker_log_prob)
            memory[4].append(listener_action)
            memory[5].append(listener_log_prob)
            memory[6].append(reward)
            memory[7].append(0 if done else 1)
            _, _, listener_value = ppo.model(obs, comms)
            memory[8].append(listener_value.item())
            print(comms, speaker_action)

            # Append speaker action to comms tensor
            comms = torch.cat([comms, speaker_action.unsqueeze(0)], dim=0)  # Shape: [N + 1]
            print(comms, speaker_action)
            obs = next_obs

        loss = ppo.train(memory)
        pbar.set_description(f"Episode: {episode + 1}, Loss: {loss:.3f}")

    ppo.save("model.pth")

def test(env, model, n_episodes=10):
    comm_map = "abcdefghijklmnopqrstuvwxyz"
    obs_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        comms = []
        comm_string = ""

        while not done:
            speaker_logits, listener_logits, _ = model(obs, comms)
            speaker_action = torch.argmax(torch.softmax(speaker_logits, dim=-1)).item()
            comm_string += comm_map[speaker_action]
            comms.append(torch.tensor([speaker_action]))

            listener_action = torch.argmax(torch.softmax(listener_logits, dim=-1)).item()
            actions = {"speaker": torch.tensor(speaker_action), "listener": torch.tensor(listener_action)}
            obs, reward, done, _ = env.step(actions)

            print(
                f"Episode: {episode + 1}, Communication: {comm_string}, "
                f"Target: {obs_map[obs.argmax().item()]}, Prediction: {obs_map[listener_action]}"
            )

import os

if os.path.exists("model.pth"):
    ppo.load("model.pth")
    test(env, model)
else:
    train(ppo, env)
    test(env, model)