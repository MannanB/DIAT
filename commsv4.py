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

# ---------------------
# Hyperparameters
# ---------------------
MAX_SEQ_LEN = 10  # Maximum length of communication sequence
OBS_SIZE = 26    # Number of possible observations
VOCAB_SIZE = 3    # Size of "language" for the speaker
MAX_TURNS = 10     # Max # of turns (also limits seq lenght)
EPISODES = 50000
MINI_EPISODES = 4
LR = 1e-4
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.1
ENTROPY_COEF = 0.05

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
    def __init__(self, vocab_size, obs_size, d_model=64, num_heads=4, ff_hidden=128, dropout=0.1):
        super(ListenerNet, self).__init__()
        self.d_model = d_model
        self.transformer = TransformerEncoder(
            vocab_size, d_model, num_heads, ff_hidden, dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, obs_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, pad_mask=None):
        """
        x shape: [batch_size, seq_len] of comm tokens in [0..vocab_size-1].
        pad_mask shape: [batch_size, seq_len].
        """
        out = self.transformer(x, src_key_padding_mask=pad_mask)
        pooled = out[:, -1, :]
        hidden = self.fc(pooled)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value

# ---------------------
# Single Model that holds Speaker & Listener
# ---------------------
class SpeakerListenerModel(nn.Module):
    def __init__(self, obs_size, vocab_size):
        super(SpeakerListenerModel, self).__init__()
        self.speaker_net = SpeakerNet(obs_size, vocab_size)
        self.listener_net = ListenerNet(vocab_size, obs_size)

    def forward_speaker(self, x, pad_mask=None):
        return self.speaker_net(x, pad_mask)

    def forward_listener(self, x, pad_mask=None):
        return self.listener_net(x, pad_mask)

# ---------------------
# Single PPO that updates both Speaker & Listener
# ---------------------
class SpeakerListenerPPO:
    def __init__(self, obs_size, vocab_size, gamma=GAMMA, eps_clip=EPS_CLIP, lr=LR,
                 lmbda=LAMBDA, entropy_coef=ENTROPY_COEF):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef

        self.policy = SpeakerListenerModel(obs_size, vocab_size)
        self.old_policy = SpeakerListenerModel(obs_size, vocab_size)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-4)

    def select_action(self, obs_tokens, comm_tokens):
        """
        obs_tokens: [batch_size, seq_len_obs], but we have 1 environment => shape (1, seq_len_obs)
        comm_tokens: [batch_size, seq_len_comm]
        
        We'll produce:
          speaker_action: next token from speaker
          listener_action: guessed obs from listener
        """
        # 1) Speaker forward
        speaker_logits, speaker_value = self.old_policy.forward_speaker(obs_tokens)
        speaker_probs = torch.softmax(speaker_logits, dim=-1)
        speaker_dist = torch.distributions.Categorical(speaker_probs)
        speaker_action = speaker_dist.sample()
        speaker_logprob = speaker_dist.log_prob(speaker_action)

        # 2) Append speaker_action to the communication tokens
        updated_comm_tokens = torch.cat(
            [comm_tokens, speaker_action.unsqueeze(-1)], dim=1
        )

        # 3) Listener forward
        listener_logits, listener_value = self.old_policy.forward_listener(updated_comm_tokens)
        listener_probs = torch.softmax(listener_logits, dim=-1)
        listener_dist = torch.distributions.Categorical(listener_probs)
        listener_action = listener_dist.sample()
        listener_logprob = listener_dist.log_prob(listener_action)

        return (speaker_action, speaker_logprob, speaker_value,
                listener_action, listener_logprob, listener_value)

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

        
        if rewards.std() > 1e-8:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards = rewards - rewards.mean()

        # Next value estimates (last step)
        next_value_speaker = 0.0
        next_value_listener = 0.0

        # Compute advantages & returns
        advantages_speaker, returns_speaker = self.compute_gae(rewards, masks, speaker_values, next_value_speaker)
        advantages_listener, returns_listener = self.compute_gae(rewards, masks, listener_values, next_value_listener)

        # For multiple minibatch updates
        for _ in range(4):
            # Re-run forward pass with current policy
            # Each time step is separate, so we'll process them in a batch.
            # First, pad obs_tokens & comm_tokens to a uniform shape: [T, <=MAX_SEQ_LEN].
            # For simplicity below, let's do them step by step in a loop or just assume T=1. 
            # In a real training scenario, you'd properly batch/pad them. 
            # Here, we'll just process each step as its own "mini-batch".
            
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

                # (Optionally) create pad masks if your sequences are shorter than MAX_SEQ_LEN
                # For example, if your comm_t is length L < MAX_SEQ_LEN:
                #   pad_length = MAX_SEQ_LEN - L
                #   padded_comm = F.pad(comm_t, (0,pad_length), value=0)
                #   pad_mask = torch.zeros_like(padded_comm).bool()
                #   pad_mask[:, L:] = True
                # In practice, you'd keep track of which tokens are real vs pad.
                # For brevity, we won't show that detail for every step here.

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
                listener_logits, listener_val = self.policy.forward_listener(updated_comm, pad_mask=None)
                listener_probs = torch.softmax(listener_logits, dim=-1)
                listener_dist = torch.distributions.Categorical(listener_probs)
                new_listener_logprob = listener_dist.log_prob(listener_actions[t].unsqueeze(0))
                listener_entropy = listener_dist.entropy().mean()

                # Compute ratio
                ratio_speaker = torch.exp(new_speaker_logprob - speaker_logprobs[t])
                ratio_listener = torch.exp(new_listener_logprob - listener_logprobs[t])

                adv_speaker = advantages_speaker[t]
                adv_listener = advantages_listener[t]

                # Surrogate objectives
                surr1_speaker = ratio_speaker * adv_speaker
                surr2_speaker = torch.clamp(ratio_speaker, 1 - self.eps_clip, 1 + self.eps_clip) * adv_speaker
                actor_loss_speaker = -torch.min(surr1_speaker, surr2_speaker)

                surr1_listener = ratio_listener * adv_listener
                surr2_listener = torch.clamp(ratio_listener, 1 - self.eps_clip, 1 + self.eps_clip) * adv_listener
                actor_loss_listener = -torch.min(surr1_listener, surr2_listener)

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
class CommEnv:
    def __init__(self, obs_size, ts_limit=10):
        self.ts_limit = ts_limit
        self.obs_size = obs_size
        # self.comms = []
        self.current_obs = None
        self.reset()

    def reset(self, obj=None):
        if obj is None:
            obj = random.randint(0, self.obs_size - 1)
        self.current_obs = torch.tensor([obj], dtype=torch.long)  # shape [1]
        self.comms = []
        self.timesteps = 0
        return self.current_obs

    def step(self, speaker_action, listener_action):
        self.timesteps += 1
        done = self.timesteps >= self.ts_limit
        if done:
            return self.current_obs, -2, True, {}
        
        # self.comms.append(speaker_action)

        # Reward: +1 if listener action matches the actual observation
        if listener_action == self.current_obs.item():
            reward = 1.0
            done = True
        else:
            reward = -(self.timesteps / self.ts_limit)

        # penalize spamming one letter
        # if any one letter is > 75% of comms and comms is more than 1, then penalize
        # if len(self.comms) > 1:
        #     counts = defaultdict(int)
        #     for c in self.comms:
        #         counts[c] += 1
        #     if max(counts.values()) > 0.75 * len(self.comms):
        #         reward -= 0.5

        return self.current_obs, reward, done, {}

# ---------------------
# Training
# ---------------------
def train(env, agent, n_episodes=EPISODES, n_mini_episodes=MINI_EPISODES):
    pbar = tqdm(range(n_episodes))
    reward_history = []
    loss_history = []

    speaker_probs_history = []

    # env.ts_limit = 1

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
            obs_tokens = obs_tokens.unsqueeze(0)  # [1, 1]
            # concatenate obs and comms
            obs_tokens = torch.concat((obs_tokens, comm_tokens), dim=-1)

            ep_reward = 0

            while not done:
                # Let agent choose speaker_action, listener_action
                (spk_a, spk_lp, spk_val,
                lis_a, lis_lp, lis_val) = agent.select_action(obs_tokens, comm_tokens)
                
                with torch.no_grad():
                    speaker_logits, _ = agent.old_policy.forward_speaker(obs_tokens)
                    spk_probs = torch.softmax(speaker_logits, dim=-1)  # shape [1, vocab_size]
                    speaker_probs_history.append(spk_probs.squeeze(0).cpu().numpy())

                # Step in the environment
                next_obs, reward, done, _ = env.step(spk_a.item(), lis_a.item())

                ep_reward += reward

                # Store
                memory['obs_tokens'].append(obs_tokens.squeeze(0).tolist())       # shape [1, obs_seq_len]
                memory['comm_tokens'].append(comm_tokens.squeeze(0).tolist())     # shape [1, comm_seq_len]
                memory['speaker_actions'].append(spk_a.item())
                memory['speaker_logprobs'].append(spk_lp.item())
                memory['speaker_values'].append(spk_val.item())
                memory['listener_actions'].append(lis_a.item())
                memory['listener_logprobs'].append(lis_lp.item())
                memory['listener_values'].append(lis_val.item())
                memory['rewards'].append(reward)
                memory['masks'].append(0 if done else 1)

                # Update comm_tokens by appending the speaker action
                comm_tokens = torch.cat([comm_tokens, spk_a.view(1,1)], dim=1)  # shape [1, comm_seq_len+1]

                # next obs
                obs = next_obs
                obs_tokens = (obs + VOCAB_SIZE).unsqueeze(0)  # shape [1,1]
                obs_tokens = torch.concat((obs_tokens, comm_tokens), dim=-1)

        # Train using the entire episode
        loss = agent.train(memory)
        loss_history.append(loss)
        reward_history.append(ep_reward)

        # balance ts limit increase in such a way that the last 10% of training is with max ts size
        # env.ts_limit = min(MAX_TURNS, int(MAX_TURNS * (ep / n_episodes) + 1))

        pbar.set_description(f"Ep {ep} Reward: {ep_reward:.2f}, Loss: {loss}, TSLIMIT: {env.ts_limit}")
        if ep % 1000 == 0:
            agent.save_to(f"checkpoint{ep}.pth")
            with open("losses.pkl", "wb") as f:
                pickle.dump(loss_history, f)
            with open("rewards.pkl", "wb") as f:
                pickle.dump(reward_history, f)


    return reward_history, loss_history, speaker_probs_history

# ---------------------
# Testing
# ---------------------
def test(env, agent, n_episodes=10):
    comm_map = "abcdefghijklmnopqrstuvwxyz"  # to visualize tokens 0->a,1->b,...
    obs_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"    # to visualize obs 0->A,1->B,...
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        comm_tokens = torch.zeros((1,0), dtype=torch.long)
        obs_tokens = (obs + VOCAB_SIZE).unsqueeze(0)  # shape [1,1]
        obs_tokens = torch.concat((obs_tokens, comm_tokens), dim=-1)

        comm_string = ""
        while not done:
            # We'll do a deterministic action by argmax
            with torch.no_grad():
                spk_logits, _ = agent.policy.forward_speaker(obs_tokens)
                spk_probs = torch.softmax(spk_logits, dim=-1)
                
                spk_a = spk_probs.argmax(dim=-1)  # [1]
            comm_string += comm_map[spk_a.item()]
            comm_tokens = torch.cat([comm_tokens, spk_a.view(1,1)], dim=1)

            with torch.no_grad():
                lis_logits, _ = agent.policy.forward_listener(comm_tokens)
                lis_probs = torch.softmax(lis_logits, dim=-1)
                lis_a = lis_probs.argmax(dim=-1)

            next_obs, reward, done, _ = env.step(spk_a.item(), lis_a.item())

            obs = next_obs
            obs_tokens = (obs + VOCAB_SIZE).unsqueeze(0)  # shape [1,1]
            obs_tokens = torch.concat((obs_tokens, comm_tokens), dim=-1)
            
            print(f"Ep: {ep}, Comm: {comm_string}, Target: {obs_map[next_obs.item()]}, Pred: {obs_map[lis_a.item()]}")

def get_translations(env, agent, n_episodes=10):
    comm_map = "abcdefghijklmnopqrstuvwxyz"  # to visualize tokens 0->a,1->b,...
    obs_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"    # to visualize obs 0->A,1->B,...
    translations = {}

    for ep in range(n_episodes):
        for obsi in range(OBS_SIZE):
            obs = env.reset(obsi)
            
            done = False
            comm_tokens = torch.zeros((1,0), dtype=torch.long)
            obs_tokens = (obs + VOCAB_SIZE).unsqueeze(0)  # shape [1,1]
            obs_tokens = torch.concat((obs_tokens, comm_tokens), dim=-1)

            comm_string = ""
            while not done:
                # We'll do a deterministic action by argmax
                with torch.no_grad():
                    spk_logits, _ = agent.policy.forward_speaker(obs_tokens)
                    spk_probs = torch.softmax(spk_logits, dim=-1)
                    
                    spk_a = spk_probs.argmax(dim=-1)  # [1]
                comm_string += comm_map[spk_a.item()]
                comm_tokens = torch.cat([comm_tokens, spk_a.view(1,1)], dim=1)

                with torch.no_grad():
                    lis_logits, _ = agent.policy.forward_listener(comm_tokens)
                    lis_probs = torch.softmax(lis_logits, dim=-1)
                    lis_a = lis_probs.argmax(dim=-1)

                next_obs, reward, done, _ = env.step(spk_a.item(), lis_a.item())

                obs = next_obs
                obs_tokens = (obs + VOCAB_SIZE).unsqueeze(0)  # shape [1,1]
                obs_tokens = torch.concat((obs_tokens, comm_tokens), dim=-1)

            if obs_map[next_obs.item()] not in translations:
                translations[obs_map[next_obs.item()]] = []
            if reward > 0:
                translations[obs_map[next_obs.item()]].append(comm_string)
    return translations

# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    env = CommEnv(OBS_SIZE, ts_limit=MAX_TURNS)
    agent = SpeakerListenerPPO(obs_size=OBS_SIZE, vocab_size=VOCAB_SIZE)

    # if not os.path.exists("speaker_listener.pth"):
    #     rewards, losses, dists = train(env, agent, n_episodes=EPISODES)
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

    # # else:
    agent.load_from("checkpoint4000.pth")

    # Test
    #test(env, agent, n_episodes=5)
    ts = get_translations(env, agent, n_episodes=10)
    print(ts)