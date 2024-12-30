import gym
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import os, pickle
import numpy as np
import matplotlib.pyplot as plt


# ---------------------
# Hyperparameters
# ---------------------

OBS_DIM = 3    # Side length of observation grid
OBS_SIZE = 3 # 0 + 2 colors
VOCAB_SIZE = 6    # Size of "language" for the speaker
MAX_TURNS = 4     # Max # of turns (also limits seq lenght)
MAX_SEQ_LEN = OBS_DIM * OBS_DIM + MAX_TURNS  # Max sequence length for speaker
EPISODES = 50000
MINI_EPISODES = 10
LR = 3e-5
GAMMA = 0.95
LAMBDA = 0.95
EPS_CLIP = 0.15
ENTROPY_COEF = 0.07

Tshape = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [1, 1, 1]
])

square = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

cross_diagonal = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

cross = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

line_vertical = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
])

line_horizontal = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
])

shapes = [Tshape, square, cross_diagonal, cross, line_vertical, line_horizontal]


# Things to try:
# multiple episodes before ppo update
# decrease lr
# balance entropy coef (0.01, 0.2)
# cirriculum learning by (increasing TS limit) OR (Slowly increasing obs space) OR (slowly increasing vocab space)

# ---------------------
# Positional Encoding
# ---------------------
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
        """x shape: [seq_len, batch_size, d_model]"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ---------------------
# Transformer Decoder
# ---------------------
class TransformerEncoder(nn.Module):
    """
    A small wrapper for a TransformerEncoder with an Embedding + PositionalEncoding.
    We allow passing a `src_key_padding_mask` for padded tokens.
    """
    def __init__(self, vocab_size, d_model, num_heads, ff_hidden_layer, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=ff_hidden_layer,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def forward(self, x, src_key_padding_mask=None):
        """
        x shape: [batch_size, seq_len]
        return shape: [batch_size, seq_len, d_model]
        """
        # Convert shape to [seq_len, batch_size]
        x = x.transpose(0, 1)  # -> [seq_len, batch_size]
        emb = self.embedding(x)  # -> [seq_len, batch_size, d_model]
        emb = self.pos_encoder(emb)  # -> [seq_len, batch_size, d_model]
        # Pass to transformer (batch_first=False)
        out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)  
        # out shape: [seq_len, batch_size, d_model]
        out = out.transpose(0, 1)  # -> [batch_size, seq_len, d_model]
        return out

# ---------------------
# Speaker & Listener Modules
# ---------------------
class SpeakerNet(nn.Module):
    """Takes an observation token plus existing comm tokens, outputs next comm token."""
    def __init__(self, obs_size, vocab_size, d_model=64, num_heads=4, ff_hidden=128, dropout=0.1):
        super(SpeakerNet, self).__init__()
        # We can define an "effective" vocab = [ real comm vocab + possible obs inputs ]
        self.effective_vocab_size = vocab_size + obs_size
        self.d_model = d_model
        
        self.transformer = TransformerEncoder(
            self.effective_vocab_size, d_model, num_heads, ff_hidden, dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, vocab_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, pad_mask=None):
        """
        x shape: [batch_size, seq_len] 
            where each token can be either a "comm token" in [0..vocab_size-1]
            or an "obs token" in [vocab_size..vocab_size+obs_size-1].
        pad_mask shape: [batch_size, seq_len] of booleans (True means "pad" / ignore).
        """
        # Convert pad_mask to shape [batch_size, seq_len] -> pass to transform as [batch_size, seq_len]
        # But the Transformer expects src_key_padding_mask: [batch_size, seq_len].
        out = self.transformer(x, src_key_padding_mask=pad_mask)  # [batch_size, seq_len, d_model]
        # Pool over seq_len dimension (simple mean)
        pooled = out[:, -1, :]
        hidden = self.fc(pooled)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value

class ListenerNet(nn.Module):
    """Takes the sequence of comm tokens, outputs a guess about the target obs."""
    def __init__(self, vocab_size, num_shapes, num_colors, d_model=64, num_heads=4, ff_hidden=128, dropout=0.1):
        super(ListenerNet, self).__init__()
        self.d_model = d_model
        self.transformer = TransformerEncoder(
            vocab_size, d_model, num_heads, ff_hidden, dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, num_shapes + num_colors)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, pad_mask=None):
        """
        x shape: [batch_size, seq_len] of comm tokens in [0..vocab_size-1].
        pad_mask shape: [batch_size, seq_len].
        """
        out = self.transformer(x, src_key_padding_mask=pad_mask)
        pooled = out[:, -1, :]
        hidden = self.fc(pooled)
        logits_combined = self.actor(hidden)
        value = self.critic(hidden)
        logits_shape, logits_color = torch.split(logits_combined, [6, 2], dim=-1)
        return logits_shape, logits_color, value
    
    def embed(self, x):
        out = self.transformer.embedding(x)
        return out

# ---------------------
# Single Model that holds Speaker & Listener
# ---------------------
class SpeakerListenerModel(nn.Module):
    def __init__(self, obs_size, vocab_size, shapes_size, colors_size):
        super(SpeakerListenerModel, self).__init__()
        self.speaker_net = SpeakerNet(obs_size, vocab_size) # note that obs_size is 2 because we have binary obs
        self.listener_net = ListenerNet(vocab_size, shapes_size, colors_size)

    def forward_speaker(self, x, pad_mask=None):
        return self.speaker_net(x, pad_mask)

    def forward_listener(self, x, pad_mask=None):
        return self.listener_net(x, pad_mask)

# ---------------------
# Single PPO that updates both Speaker & Listener
# ---------------------
class SpeakerListenerPPO:
    def __init__(self, obs_size, vocab_size, shapes_size, colors_size, gamma=GAMMA, eps_clip=EPS_CLIP, lr=LR,
                 lmbda=LAMBDA, entropy_coef=ENTROPY_COEF):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef

        self.policy = SpeakerListenerModel(obs_size, vocab_size, shapes_size, colors_size)
        self.old_policy = SpeakerListenerModel(obs_size, vocab_size, shapes_size, colors_size)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-4)

    def select_action(self, obs_shape, comm_tokens):
        # 1) Speaker forward
        speaker_logits, speaker_value = self.old_policy.forward_speaker(obs_shape)
        speaker_probs = torch.softmax(speaker_logits, dim=-1)
        speaker_dist = torch.distributions.Categorical(speaker_probs)
        speaker_action = speaker_dist.sample()
        speaker_logprob = speaker_dist.log_prob(speaker_action)

        # 2) Append speaker_action to the communication tokens
        updated_comm_tokens = torch.cat(
            [comm_tokens, speaker_action.unsqueeze(-1)], dim=1
        )

        # 3) Listener forward
        listener_logits_shape, listener_logits_color, listener_value = self.old_policy.forward_listener(updated_comm_tokens)

        # Sample actions for both shape and color
        listener_probs_shape = torch.softmax(listener_logits_shape, dim=-1)
        listener_probs_color = torch.softmax(listener_logits_color, dim=-1)

        listener_dist_shape = torch.distributions.Categorical(listener_probs_shape)
        listener_dist_color = torch.distributions.Categorical(listener_probs_color)

        listener_action_shape = listener_dist_shape.sample()
        listener_action_color = listener_dist_color.sample()

        # Compute log probabilities
        listener_logprob_shape = listener_dist_shape.log_prob(listener_action_shape)
        listener_logprob_color = listener_dist_color.log_prob(listener_action_color)

        # Sum log probabilities (for combined optimization later)
        listener_logprob = listener_logprob_shape + listener_logprob_color

        return (speaker_action, speaker_logprob, speaker_value,
                (listener_action_shape, listener_action_color),
                (listener_logprob_shape, listener_logprob_color), listener_value)

    def compute_gae(self, rewards, masks, values, next_value):
        """
        Standard GAE (Generalized Advantage Estimator).
        `values` are the predicted state values from the model (either speaker or listener).
        We assume reward is shared by both, so we'll apply the same advantage to each.
        """
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            advantages.insert(0, gae)
            next_value = values[i]
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        return advantages, returns

    def train(self, memory):
        """
        memory is a dictionary containing:
          - obs_tokens: shape [T, seq_len_obs]
          - comm_tokens: shape [T, seq_len_comm]
          - speaker_actions
          - speaker_logprobs
          - speaker_values
          - listener_actions
          - listener_logprobs
          - listener_values
          - rewards
          - masks (1 - done)
        We'll do a single PPO update for both speaker & listener.
        """
        obs_tokens = memory['obs_tokens']
        comm_tokens = memory['comm_tokens']
        speaker_actions = memory['speaker_actions']
        speaker_logprobs = memory['speaker_logprobs']
        speaker_values = memory['speaker_values']
        listener_actions = memory['listener_actions']
        listener_logprobs = memory['listener_logprobs']
        listener_values = memory['listener_values']
        rewards = memory['rewards']
        masks = memory['masks']  # 0 if done, 1 if not done

        # Convert to torch if needed
        speaker_actions = torch.LongTensor(speaker_actions)
        listener_actions = torch.LongTensor(listener_actions)
        speaker_logprobs = torch.FloatTensor(speaker_logprobs)
        listener_logprobs = torch.FloatTensor(listener_logprobs)
        speaker_values = torch.FloatTensor(speaker_values)
        listener_values = torch.FloatTensor(listener_values)
        rewards = np.array(rewards)
        masks = np.array(masks)

        listener_logprobs_shape = listener_logprobs[:, 0]
        listener_logprobs_color = listener_logprobs[:, 1]
        listener_action_shape = listener_actions[:, 0]
        listener_action_color = listener_actions[:, 1]

        # if rewards.std() > 1e-8:
        #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # else:
        #     rewards = rewards - rewards.mean()

        # Next value estimates (last step)
        next_value_speaker = 0.0
        next_value_listener = 0.0

        # Compute advantages & returns
        advantages_speaker, returns_speaker = self.compute_gae(rewards, masks, speaker_values, next_value_speaker)
        advantages_listener, returns_listener = self.compute_gae(rewards, masks, listener_values, next_value_listener)

        # For multiple minibatch updates
        for _ in range(4):
            # We'll accumulate losses across all steps, then do one optimizer step.
            total_loss = 0.0
            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_entropy = 0.0

            T = len(obs_tokens)
            for t in range(T):
                obs_t = obs_tokens[t]     # shape [1, obs_seq_len]
                comm_t = comm_tokens[t]   # shape [1, comm_seq_len]

                # Make sure they're LongTensors
                obs_t = torch.LongTensor(obs_t).unsqueeze(0)   # [1, obs_seq_len]
                comm_t = torch.LongTensor(comm_t).unsqueeze(0) # [1, comm_seq_len]

                # speaker forward
                speaker_logits, speaker_val = self.policy.forward_speaker(obs_t, pad_mask=None)
                speaker_probs = torch.softmax(speaker_logits, dim=-1)
                speaker_dist = torch.distributions.Categorical(speaker_probs)
                new_speaker_logprob = speaker_dist.log_prob(speaker_actions[t].unsqueeze(0))
                speaker_entropy = speaker_dist.entropy().mean()

                # listener forward
                # We append the *actual* speaker_actions[t] to comm_t
                # to replicate the same input the old policy had:
                updated_comm = torch.cat([comm_t, speaker_actions[t].view(1,1)], dim=1)
                listener_logits_shape, listener_logits_color, listener_val = self.policy.forward_listener(updated_comm, pad_mask=None)
                # Sample actions for both shape and color
                listener_probs_shape = torch.softmax(listener_logits_shape, dim=-1)
                listener_probs_color = torch.softmax(listener_logits_color, dim=-1)

                listener_dist_shape = torch.distributions.Categorical(listener_probs_shape)
                listener_dist_color = torch.distributions.Categorical(listener_probs_color)
                # Compute log probabilities
                new_listener_logprob_shape = listener_dist_shape.log_prob(listener_action_shape[t].unsqueeze(0))
                new_listener_logprob_color = listener_dist_color.log_prob(listener_action_color[t].unsqueeze(0))

                # Sum log probabilities (for combined optimization later)
                listener_entropy = (listener_dist_shape.entropy().mean() + listener_dist_color.entropy().mean()) / 2

                # Compute ratio
                ratio_speaker = torch.exp(new_speaker_logprob - speaker_logprobs[t])
                ratio_listener_shape = torch.exp(new_listener_logprob_shape - listener_logprobs_shape[t])
                ratio_listener_color = torch.exp(new_listener_logprob_color - listener_logprobs_color[t])

                adv_speaker = advantages_speaker[t]
                adv_listener = advantages_listener[t]

                # Surrogate objectives
                surr1_speaker = ratio_speaker * adv_speaker
                surr2_speaker = torch.clamp(ratio_speaker, 1 - self.eps_clip, 1 + self.eps_clip) * adv_speaker
                actor_loss_speaker = -torch.min(surr1_speaker, surr2_speaker)

                surr1_listener_shape = ratio_listener_shape * adv_listener
                surr2_listener_shape = torch.clamp(ratio_listener_shape, 1 - self.eps_clip, 1 + self.eps_clip) * adv_listener
                actor_loss_listener_shape = -torch.min(surr1_listener_shape, surr2_listener_shape)
                surr1_listener_color = ratio_listener_color * adv_listener
                surr2_listener_color = torch.clamp(ratio_listener_color, 1 - self.eps_clip, 1 + self.eps_clip) * adv_listener
                actor_loss_listener_color = -torch.min(surr1_listener_color, surr2_listener_color)
                actor_loss_listener = actor_loss_listener_shape + actor_loss_listener_color
                # Critic losses
                ret_speaker = returns_speaker[t]
                ret_listener = returns_listener[t]

                # We do clip value loss similar to PPO
                value_pred_speaker_clipped = speaker_values[t] + (ret_speaker - speaker_values[t]).clamp(
                    -self.eps_clip, self.eps_clip
                )
                speaker_val_loss_clipped = (value_pred_speaker_clipped - ret_speaker) ** 2
                speaker_val_loss_normal = (speaker_val - ret_speaker) ** 2
                critic_loss_speaker = 0.5 * torch.max(speaker_val_loss_clipped, speaker_val_loss_normal)

                value_pred_listener_clipped = listener_values[t] + (ret_listener - listener_values[t]).clamp(
                    -self.eps_clip, self.eps_clip
                )
                listener_val_loss_clipped = (value_pred_listener_clipped - ret_listener) ** 2
                listener_val_loss_normal = (listener_val - ret_listener) ** 2
                critic_loss_listener = 0.5 * torch.max(listener_val_loss_clipped, listener_val_loss_normal)
                # Sum up losses
                total_actor_loss += (actor_loss_speaker + actor_loss_listener)
                total_critic_loss += (critic_loss_speaker + critic_loss_listener)
                total_entropy += (speaker_entropy + listener_entropy)

            # Average out over T steps
            total_actor_loss /= T
            total_critic_loss /= T
            total_entropy /= T

            # Final PPO loss
            loss = total_actor_loss + total_critic_loss - self.entropy_coef * total_entropy

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Finally, update old_policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        return loss.item()

    def save_to(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load_from(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.old_policy.load_state_dict(self.policy.state_dict())

# ---------------------
# Simple Communication Environment
# ---------------------
class VisualCommEnv:
    def __init__(self, obs_dim, ts_limit=10):
        self.ts_limit = ts_limit
        self.obs_dim = obs_dim
        self.current_obs = None
        self.current_shape = None
        self.current_color = None
        self.reset()

    def reset(self, current_shape=None, current_color=None):
        if current_shape is None:
            current_shape_i = random.randint(0, len(shapes)-1)
        else:
            current_shape_i = current_shape

        if current_color is None:
            self.current_color = random.randint(0, 1)
        else:
            self.current_color = current_color

        self.current_shape = current_shape_i
        self.current_obs = shapes[current_shape_i]
        self.current_obs = torch.tensor(self.current_obs).view(-1) * (self.current_color + 1)
        self.comms = []
        self.timesteps = 0
        return self.current_obs

    def step(self, speaker_action, listener_action_shape, listener_action_color):
        self.timesteps += 1
        done = self.timesteps >= self.ts_limit
        if done:
            return self.current_obs, -2, True, {}
        
        # self.comms.append(speaker_action)
        # Reward: +1 if listener action matches the actual observation
        if listener_action_shape == self.current_shape and listener_action_color == self.current_color:
            reward = (self.ts_limit + 1) / 2
            done = True
        elif listener_action_shape == self.current_shape:
            reward = 1
        elif listener_action_color == self.current_color:
            reward = 0
        else:
            reward = -(self.timesteps / self.ts_limit)

        return self.current_obs, reward, done, {}

# ---------------------
# Training
# ---------------------
def train(env, agent, n_episodes=EPISODES, n_mini_episodes=MINI_EPISODES, start_point=0):
    pbar = tqdm(range(n_episodes), initial=start_point)
    reward_history = []
    loss_history = []

    speaker_probs_history = []

    for ep in pbar:
        
        # We'll store transitions for each step in this episode:
        memory = {
            'obs_tokens': [],
            'comm_tokens': [],
            'speaker_actions': [],
            'speaker_logprobs': [],
            'speaker_values': [],
            'listener_actions': [],
            'listener_logprobs': [],
            'listener_values': [],
            'rewards': [],
            'masks': []
        }
        total_ep_reward = 0
        total_correct = 0
        for _ in range(n_mini_episodes):
            obs = env.reset()
            done = False
            # Initialize comm_tokens to empty: shape [1, 0]  (no tokens yet)
            comm_tokens = torch.zeros((1, 0), dtype=torch.long)
            # The speaker's input is observation tokens. 
            # We can shift the observation token to avoid colliding with actual comm tokens 
            # if we want a single embedding space:
            # e.g. speaker sees obs as token in [vocab_size..vocab_size+obs_size-1].
            obs_tokens = obs + VOCAB_SIZE  # shift by +2 if vocab_size=2. shape [1]
            obs_tokens = obs_tokens.view(1, -1)  # [3, 3] -> [1, 9]
            # concatenate obs and comms
            obs_tokens = torch.concat((obs_tokens, comm_tokens), dim=-1)

            while not done:
                # Let agent choose speaker_action, listener_action
                (spk_a, spk_lp, spk_val,
                lis_a, lis_lp, lis_val) = agent.select_action(obs_tokens, comm_tokens)

                # Step in the environment
                next_obs, reward, done, _ = env.step(spk_a.item(), lis_a[0].item(), lis_a[1].item())

                total_ep_reward += reward

                # Store
                memory['obs_tokens'].append(obs_tokens.squeeze(0).tolist())       # shape [1, obs_seq_len]
                memory['comm_tokens'].append(comm_tokens.squeeze(0).tolist())     # shape [1, comm_seq_len]
                memory['speaker_actions'].append(spk_a.item())
                memory['speaker_logprobs'].append(spk_lp.item())
                memory['speaker_values'].append(spk_val.item())
                memory['listener_actions'].append(lis_a)
                memory['listener_logprobs'].append(lis_lp)
                memory['listener_values'].append(lis_val.item())
                memory['rewards'].append(reward)
                memory['masks'].append(0 if done else 1)

                # Update comm_tokens by appending the speaker action
                comm_tokens = torch.cat([comm_tokens, spk_a.view(1,1)], dim=1)  # shape [1, comm_seq_len+1]

                # next obs
                obs = next_obs
                obs_tokens = obs + VOCAB_SIZE  # shift by +2 if vocab_size=2. shape [1]
                obs_tokens = obs_tokens.view(1, -1)  # [3, 3] -> [1, 9]
                # concatenate obs and comms
                obs_tokens = torch.concat((obs_tokens, comm_tokens), dim=-1)
            # check if the last action was correct
            if env.current_shape == lis_a[0].item() and env.current_color == lis_a[1].item():
                total_correct += 1

        total_ep_reward /= n_mini_episodes
        total_correct /= n_mini_episodes
        # Train using the entire episode
        loss = agent.train(memory)
        loss_history.append(loss)
        reward_history.append(total_ep_reward)

        # balance ts limit increase in such a way that the last 10% of training is with max ts size
        # env.ts_limit = min(MAX_TURNS, int(MAX_TURNS * (ep / n_episodes) + 1))

        pbar.set_description(f"Ep {ep} Reward: {total_ep_reward:.2f}, Loss: {loss}, Acc: {total_correct:.2f}")
        if ep % 1000 == 0:
            agent.save_to(f"checkpoint{ep + start_point}.pth")
            with open("losses.pkl", "wb") as f:
                pickle.dump(loss_history, f)
            with open("rewards.pkl", "wb") as f:
                pickle.dump(reward_history, f)


    return reward_history, loss_history, speaker_probs_history

# ---------------------
# Testing
# ---------------------
def test(env, agent, n_episodes=10):
    num_correct = 0
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        comm_tokens = torch.zeros((1, 0), dtype=torch.long)
        obs_tokens = obs + VOCAB_SIZE
        obs_tokens = obs_tokens.view(1, -1)
        obs_tokens = torch.cat((obs_tokens, comm_tokens), dim=-1)
        ep_reward = 0

        while not done:
            (spk_a, spk_lp, spk_val,
            lis_a, lis_lp, lis_val) = agent.select_action(obs_tokens, comm_tokens)

            next_obs, reward, done, _ = env.step(spk_a.item(), lis_a[0].item(), lis_a[1].item())        
            ep_reward += reward
            comm_tokens = torch.cat([comm_tokens, spk_a.view(1,1)], dim=1)
            obs = next_obs
            obs_tokens = obs + VOCAB_SIZE
            obs_tokens = obs_tokens.view(1, -1)
            obs_tokens = torch.cat((obs_tokens, comm_tokens), dim=-1)

        if env.current_shape == lis_a[0].item() and env.current_color == lis_a[1].item():
            num_correct += 1

    # print(f"Accuracy: {num_correct}/{n_episodes}")
    return num_correct / n_episodes
    

def get_translations(env, agent, n_episodes=10, only_color=False, only_shape=False):
    comm_map = "abcdefghijklmnopqrstuvwxyz"
    shapes_map = ["Tshape", "square", "cross_diagonal", "cross", "line_vertical", "line_horizontal"]
    num_correct = 0
    translations = {}
    for ep in range(n_episodes):
        for color in range(2):
            for shape in range(len(shapes)):
                obs = env.reset(current_shape=shape, current_color=color)
                done = False
                comm_tokens = torch.zeros((1, 0), dtype=torch.long)
                obs_tokens = obs + VOCAB_SIZE
                obs_tokens = obs_tokens.view(1, -1)
                
                obs_tokens = torch.cat((obs_tokens, comm_tokens), dim=-1)
                comm_str = ""
                ep_reward = 0   

                while not done:
                    (spk_a, spk_lp, spk_val,
                    lis_a, lis_lp, lis_val) = agent.select_action(obs_tokens, comm_tokens)
                    next_obs, reward, done, _ = env.step(spk_a.item(), lis_a[0].item(), lis_a[1].item())        
                    ep_reward += reward
                    comm_str += comm_map[spk_a.item()]
                    comm_tokens = torch.cat([comm_tokens, spk_a.view(1,1)], dim=1)
                    obs = next_obs
                    obs_tokens = obs + VOCAB_SIZE
                    obs_tokens = obs_tokens.view(1, -1)
                    obs_tokens = torch.cat((obs_tokens, comm_tokens), dim=-1)
                if only_color:
                    key = ("red" if color == 0 else "blue")
                elif only_shape:
                    key = shapes_map[shape]
                else:
                    key = (shapes_map[shape], "red" if color == 0 else "blue")
                if key not in translations:
                    translations[key] = set()
                if env.current_color == lis_a[1].item() and only_color:
                    translations[key].add(comm_str)
                    num_correct += 1
                elif env.current_shape == lis_a[0].item() and only_shape:
                    translations[key].add(comm_str)
                    num_correct += 1
                elif env.current_shape == lis_a[0].item() and env.current_color == lis_a[1].item():
                    translations[key].add(comm_str)
                    num_correct += 1
    print(f"Accuracy: {num_correct}/{n_episodes*len(shapes)*2}")
    return translations

def TNSEProjection(agent):
    # project vocab to 2D using TSNE for only the listener embeddings

    # get all the embeddings
    embeddings_listener = []
    embeddings_speaker = []
    for i in range(VOCAB_SIZE):
        comm_tokens = torch.tensor([i]).view(1, 1)
        embedding_listener = agent.policy.listener_net.embed(comm_tokens)
        embeddings_listener.append(embedding_listener.squeeze(0).detach().numpy())

    for i in range(VOCAB_SIZE + OBS_SIZE):
        comm_tokens = torch.tensor([i]).view(1, 1)
        embedding_speaker = agent.policy.speaker_net.transformer.embedding(comm_tokens)
        embeddings_speaker.append(embedding_speaker.squeeze(0).detach().numpy())

    embeddings_listener = np.array(embeddings_listener).squeeze(1)
    embeddings_speaker = np.array(embeddings_speaker).squeeze(1)
    # project to 2D
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0, perplexity=1)
    embeddings_2d_listener = tsne.fit_transform(embeddings_listener)
    embeddings_2d_speaker = tsne.fit_transform(embeddings_speaker)

    # I want to display figures side by side and label them

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    vocab_map = "abcdefghijklmnopqrstuvwxyz"[:VOCAB_SIZE]
    for i, txt in enumerate(range(VOCAB_SIZE)):
        axs[0].annotate(vocab_map[i], (embeddings_2d_listener[i, 0], embeddings_2d_listener[i, 1]))
    for i, txt in enumerate(range(VOCAB_SIZE + OBS_SIZE)):
        if i < VOCAB_SIZE:
            axs[1].annotate(vocab_map[i], (embeddings_2d_speaker[i, 0], embeddings_2d_speaker[i, 1]))
        else:
            axs[1].annotate(["None", "Red", "Blue"][i - VOCAB_SIZE], (embeddings_2d_speaker[i, 0], embeddings_2d_speaker[i, 1]))
    axs[0].scatter(embeddings_2d_listener[:, 0], embeddings_2d_listener[:, 1])
    axs[0].set_title("Listener Embeddings")
    axs[1].scatter(embeddings_2d_speaker[:, 0], embeddings_2d_speaker[:, 1])
    axs[1].set_title("Speaker Embeddings")
    plt.show()

    embeddings_shapes_blue = []
    embeddings_shapes_red = []
    for i in range(len(shapes)):
        comm_tokens = torch.tensor(shapes[i]).view(1, -1) + VOCAB_SIZE
        embedding = agent.policy.speaker_net.transformer.embedding(comm_tokens).mean(dim=1)
        embeddings_shapes_red.append(embedding.squeeze(0).detach().numpy())
        comm_tokens = torch.tensor(shapes[i]).view(1, -1) * 2 + VOCAB_SIZE
        embedding = agent.policy.speaker_net.transformer.embedding(comm_tokens).mean(dim=1)
        embeddings_shapes_blue.append(embedding.squeeze(0).detach().numpy())
    embeddings_shapes_red = np.array(embeddings_shapes_red)
    embeddings_shapes_blue = np.array(embeddings_shapes_blue)
    
    all_embeddings = np.concatenate([embeddings_shapes_red, embeddings_shapes_blue])
    # project to 2D
    embeddings_2d_shapes = tsne.fit_transform(all_embeddings)

    fig = plt.figure(figsize=(6, 5))

    shapes_map = ["Tshape", "square", "cross_diagonal", "cross", "line_vertical", "line_horizontal"]

    for i, txt in enumerate(range(len(shapes))):
        plt.annotate(shapes_map[i] + " red", (embeddings_2d_shapes[i, 0], embeddings_2d_shapes[i, 1]))
    for i, txt in enumerate(range(len(shapes))):
        plt.annotate(shapes_map[i] + " blue", (embeddings_2d_shapes[i + len(shapes), 0], embeddings_2d_shapes[i + len(shapes), 1]))
    plt.scatter(embeddings_2d_shapes[:, 0], embeddings_2d_shapes[:, 1])
    plt.title("Shape Embeddings")
    plt.show()


    # plot
    # plt.figure(figsize=(6, 5))
    # plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    # vocab_map = "abcdefghijklmnopqrstuvwxyz"
    # for i, txt in enumerate(range(VOCAB_SIZE)):
    #     plt.annotate(vocab_map[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    # plt.show()

    # return embeddings_2d

def display_all_shapes():
    shapes_map = ["Tshape", "square", "cross_diagonal", "cross", "line_vertical", "line_horizontal"]

    # display as figs, in binary, and label them
    # dont show axis
    fig, axs = plt.subplots(1, 6, figsize=(12, 8))
    for i in range(len(shapes)):
        ax = axs[i]
        # invert the colors
        ax.imshow(shapes[i].reshape(3, 3), cmap="gray_r")
        ax.axis("off")
        ax.set_title(shapes_map[i])
    plt.show()

def create_accuracy_graph(env, folder_path):
    import matplotlib.pyplot as plt
    import pickle
    import os

    acc = []
    episodes = []
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith(".pth") and "listener" not in file and "Best" not in file:
            agent = SpeakerListenerPPO(obs_size=OBS_SIZE, vocab_size=VOCAB_SIZE, colors_size=OBS_SIZE-1, shapes_size=len(shapes))
            agent.load_from(os.path.join(folder_path, file))
            acci = test(env, agent, n_episodes=30)
            acc.append(acci)
            episodes.append(int(file.split(".")[0][10:]))
    acc_sorted = sorted(zip(episodes, acc), key=lambda x: x[0])
    episodes, acc = zip(*acc_sorted)
    plt.plot(episodes, acc)
    plt.title("Accuracy")
    plt.xlabel("Episodes")
    plt.ylabel("Accuracy")
    plt.show()


# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    display_all_shapes()
    env = VisualCommEnv(OBS_DIM, ts_limit=MAX_TURNS)

    # create_accuracy_graph(env, "./saves/visual2")

    agent = SpeakerListenerPPO(obs_size=OBS_SIZE, vocab_size=VOCAB_SIZE, colors_size=OBS_SIZE-1, shapes_size=len(shapes))

    agent.load_from("./saves/visual2/checkpoint27000.pth")
    # for i in range(EPISODES, -1000, -1000):
    #     if os.path.exists(f"checkpoint{i}.pth"):
    #         agent.load_from(f"checkpoint{i}.pth")
    #         print(f"Loaded from checkpoint{i}.pth")
    #         break

    # if not os.path.exists("speaker_listener.pth"):
    #     rewards, losses, dists = train(env, agent, n_episodes=EPISODES, start_point=i)
    #     agent.save_to("speaker_listener.pth")

    #     # Plot if desired
    #     import matplotlib.pyplot as plt
    #     import pickle
    #     # save losses
    #     with open("losses.pkl", "wb") as f:
    #         pickle.dump(losses, f)
    #     plt.plot(losses)
    #     plt.title("Losses")
    #     plt.show()
    #     # save losses
    #     with open("rewards.pkl", "wb") as f:
    #         pickle.dump(rewards, f)
    #     plt.plot(rewards)
    #     plt.title("Rewards")
    #     plt.show()


    #     speaker_probs_history = np.array(dists)  # shape [T, vocab_size]
    #     for token_id in range(VOCAB_SIZE):
    #         plt.plot(speaker_probs_history[:, token_id], label=f"Token {token_id}")
    #     plt.legend()
    #     plt.show()



    # Test
    test(env, agent, n_episodes=100)
    print("Translations")
    print("Color")
    translations = get_translations(env, agent, n_episodes=3, only_color=True, only_shape=False)
    print(translations)
    print("Shape")
    translations = get_translations(env, agent, n_episodes=5, only_color=False, only_shape=True)
    print(translations)
    print("Both")
    translations = get_translations(env, agent, n_episodes=5, only_color=False, only_shape=False)
    print(translations)
    TNSEProjection(agent)