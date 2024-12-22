import gym
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# ==============================
#  1. Define model components
# ==============================

def make_padding_mask(x: torch.LongTensor, pad_token: int) -> torch.BoolTensor:
    """
    x: [seq_len, batch_size] of token IDs
    Returns: mask: [batch_size, seq_len], with True in positions that
             should be masked (i.e. PAD) and False otherwise.
    """
    # True where x == pad_token; then transpose to [batch_size, seq_len]
    return (x == pad_token).transpose(0, 1)

# Define PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

# Define TransformerDecoder
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_hidden_layer, dropout):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_hidden_layer,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x

# Define ActorCriticListener
class ActorCriticListener(nn.Module):
    def __init__(self, vocab_size, actions, pad_token=None):
        super(ActorCriticListener, self).__init__()
        self.d_model = 64
        self.heads = 4
        self.pad_token = pad_token
        self.transformer = TransformerDecoder(vocab_size + 1, self.d_model, self.heads, 128, 0.1)
        self.total_vocab_size = vocab_size
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        """
        x shape: [sequence_length, batch_size]
        We assume a single sequence in batch_size=1 scenario.
        """
        # replace all PAD_TOKEN with the last token in the vocab

                
        if len(x.shape) == 1:  # unbatched
            x = x.unsqueeze(0)
        
        if self.pad_token is not None and x[x == self.pad_token].size(0) > 0:
            pad_mask = make_padding_mask(x, self.pad_token)  # shape [1, sequence_length]
        else:
            pad_mask = None
        x[x == self.pad_token] = self.total_vocab_size


        x = self.transformer(x, src_key_padding_mask=pad_mask)
        # For simplicity, let's just average over the sequence dimension
        x = x.mean(dim=1)
        x = self.fc(x)
        logits = self.actor(x).squeeze()
        value = self.critic(x)
        return logits, value

# Define ActorCriticSpeaker
class ActorCriticSpeaker(nn.Module):
    def __init__(self, input_size, vocab_size, pad_token=None):
        super(ActorCriticSpeaker, self).__init__()
        self.d_model = 64
        self.heads = 4
        self.pad_token = pad_token
        self.transformer = TransformerDecoder(
            vocab_size + input_size + 1, self.d_model, self.heads, 128, 0.1
        )
        self.total_vocab_size = vocab_size + input_size
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, vocab_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        """
        x shape: [sequence_length, batch_size]
        We assume a single sequence in batch_size=1 scenario.
        """
        if len(x.shape) == 1:  # unbatched
            x = x.unsqueeze(0)

        # print("INPUT FOR SPEAKER", x)
        # replace all PAD_TOKEN with the last token in the vocab
        
        if self.pad_token is not None and x[x == self.pad_token].size(0) > 0:
            pad_mask = make_padding_mask(x, self.pad_token)  # shape [1, sequence_length]
        else:
            pad_mask = None
        x[x == self.pad_token] = self.total_vocab_size
        # print(x, self.pad_token, self.total_vocab_size)



        # print("FORWARD SHAPE FOR SPEAKER:", x.shape)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        # print("AFTER TRANSFORMER SHAPE FOR SPEAKER:", x.shape)
        # For simplicity, let's just average over the sequence dimension
        x = x.mean(dim=1)
        # print("AFTER SQUEEZE SHAPE FOR SPEAKER:", x.shape)
        x = self.fc(x)
        # print("AFTER FC SHAPE FOR SPEAKER:", x.shape)
        logits = self.actor(x).squeeze()
        # print("AFTER ACTOR SHAPE FOR SPEAKER:", logits.shape)
        value = self.critic(x)
        return logits, value


# ==========================================
#  2. Single PPO for both Speaker & Listener
# ==========================================
class MultiAgentPPO:
    def __init__(
        self,
        obs_size,
        language_size,
        gamma=0.99,
        eps_clip=0.1,
        lr=1e-4,
        lmbda=0.95,
        entropy_coef=0.05,
    ):
        """
        obs_size: dimension of 'world' observation (for the listener's action space)
        language_size: dimension of the symbolic language (for the speaker's vocab)
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef

        # Instantiate Speaker & Listener
        self.speaker = ActorCriticSpeaker(obs_size, language_size, pad_token=-1)
        self.listener = ActorCriticListener(language_size, obs_size, pad_token=-1)

        # Old networks (for ratio-based PPO)
        self.old_speaker = ActorCriticSpeaker(obs_size, language_size, pad_token=-1)
        self.old_listener = ActorCriticListener(language_size, obs_size, pad_token=-1)
        self.old_speaker.load_state_dict(self.speaker.state_dict())
        self.old_listener.load_state_dict(self.listener.state_dict())

        # Single optimizer for both
        self.optimizer = optim.Adam(
            list(self.speaker.parameters()) + list(self.listener.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )

    def select_speaker_action(self, obs, deterministic=False):
        """Select action for speaker using old_speaker policy."""
        obs = self.convert_obs(obs)
        logits, _ = self.old_speaker(obs)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)

        if deterministic:
            action = probs.argmax()
            # print(probs, action)
            log_prob = torch.log(probs[action])
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob

    def select_listener_action(self, obs, deterministic=False):
        """Select action for listener using old_listener policy."""
        obs = self.convert_obs(obs)
        logits, _ = self.old_listener(obs)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)

        if deterministic:
            action = probs.argmax()
            # print(probs, action)
            log_prob = torch.log(probs[action])
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob

    def compute_gae(self, rewards, masks, values, next_value):
        """
        Standard GAE (Generalized Advantage Estimator).
        """
        advantages = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            advantages.insert(0, gae)
            next_value = values[i]
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        return advantages, returns
        
    def convert_obs(self, obs):
        """Utility to handle list vs tensor conversions."""
        # handle padding
        if isinstance(obs, list):
            obs = [pad_sequence(o) for o in obs]
            obs = torch.stack(obs)
            return obs
        else:
            # print("asdf", obs)
            return pad_sequence(obs)

    def train(self, memory_speaker, memory_listener, optimize=True, num_updates=4):
        """
        Perform a single PPO update (but we do multiple epochs).
        We'll do a single backward() that sums Speaker and Listener losses.
        """

        # Unpack memory for speaker
        s_obs, s_actions, s_log_probs, s_rewards, s_masks, s_values = memory_speaker
        s_obs = self.convert_obs(s_obs)
        s_actions = torch.LongTensor(s_actions)
        s_log_probs = torch.FloatTensor(s_log_probs).detach()
        s_values = torch.FloatTensor(s_values).detach()

        # Unpack memory for listener
        l_obs, l_actions, l_log_probs, l_rewards, l_masks, l_values = memory_listener
        l_obs = self.convert_obs(l_obs)
        l_actions = torch.LongTensor(l_actions)
        l_log_probs = torch.FloatTensor(l_log_probs).detach()
        l_values = torch.FloatTensor(l_values).detach()

        # Normalize rewards to help with training stability (optional)
        s_rewards = np.array(s_rewards)
        if s_rewards.std() > 1e-8:
            s_rewards = (s_rewards - s_rewards.mean()) / (s_rewards.std() + 1e-8)
        else:
            s_rewards = s_rewards - s_rewards.mean()

        l_rewards = np.array(l_rewards)
        if l_rewards.std() > 1e-8:
            l_rewards = (l_rewards - l_rewards.mean()) / (l_rewards.std() + 1e-8)
        else:
            l_rewards = l_rewards - l_rewards.mean()

        # Next state value predictions (for GAE terminal)
        next_s_value = self.speaker(s_obs[-1:])[1].item()
        next_l_value = self.listener(l_obs[-1:])[1].item()

        # Compute advantages
        s_advantages, s_returns = self.compute_gae(
            s_rewards, s_masks, s_values, next_s_value
        )
        l_advantages, l_returns = self.compute_gae(
            l_rewards, l_masks, l_values, next_l_value
        )

        # PPO update
        final_loss_val = 0.0
        for _ in range(num_updates):
            # 1) Forward pass speaker
            s_logits, s_value_preds = self.speaker(s_obs)
            s_probs = torch.softmax(s_logits, dim=-1)
            s_probs = torch.clamp(s_probs, 1e-8, 1.0 - 1e-8)
            s_dist = torch.distributions.Categorical(s_probs)
            # print(s_dist.shape, s_actions.shape)
            new_s_log_probs = s_dist.log_prob(s_actions)
            s_entropy = s_dist.entropy().mean()

            # Speaker ratio
            s_ratio = torch.exp(new_s_log_probs - s_log_probs)

            # Surrogate objectives
            s_surr1 = s_ratio * s_advantages
            s_surr2 = (
                torch.clamp(s_ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                * s_advantages
            )
            s_actor_loss = -torch.min(s_surr1, s_surr2).mean()

            # Critic loss
            s_value_pred_clipped = s_values + (s_returns - s_values).clamp(
                -self.eps_clip, self.eps_clip
            )
            s_value_loss_clipped = (s_value_pred_clipped - s_returns).pow(2)
            s_critic_loss = 0.5 * torch.max(
                s_value_loss_clipped, (s_value_preds - s_returns).pow(2)
            ).mean()

            # 2) Forward pass listener
            l_logits, l_value_preds = self.listener(l_obs)
            l_probs = torch.softmax(l_logits, dim=-1)
            l_probs = torch.clamp(l_probs, 1e-8, 1.0 - 1e-8)
            l_dist = torch.distributions.Categorical(l_probs)
            new_l_log_probs = l_dist.log_prob(l_actions)
            l_entropy = l_dist.entropy().mean()

            # Listener ratio
            l_ratio = torch.exp(new_l_log_probs - l_log_probs)

            # Surrogate objectives
            l_surr1 = l_ratio * l_advantages
            l_surr2 = (
                torch.clamp(l_ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                * l_advantages
            )
            l_actor_loss = -torch.min(l_surr1, l_surr2).mean()

            # Critic loss
            l_value_pred_clipped = l_values + (l_returns - l_values).clamp(
                -self.eps_clip, self.eps_clip
            )
            l_value_loss_clipped = (l_value_pred_clipped - l_returns).pow(2)
            l_critic_loss = 0.5 * torch.max(
                l_value_loss_clipped, (l_value_preds - l_returns).pow(2)
            ).mean()

            # Combine them all for one update
            speaker_loss = s_actor_loss + s_critic_loss - self.entropy_coef * s_entropy
            listener_loss = l_actor_loss + l_critic_loss - self.entropy_coef * l_entropy
            loss = speaker_loss + listener_loss

            if optimize:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            final_loss_val = loss.item()

        # Update old networks
        self.old_speaker.load_state_dict(self.speaker.state_dict())
        self.old_listener.load_state_dict(self.listener.state_dict())

        # Return the combined final loss and each agent's partial losses if needed
        return {
            "loss_total": final_loss_val,
            "speaker_actor": s_actor_loss.item(),
            "speaker_critic": s_critic_loss.item(),
            "listener_actor": l_actor_loss.item(),
            "listener_critic": l_critic_loss.item(),
        }

    def save_to(self, speaker_path, listener_path):
        torch.save(self.speaker.state_dict(), speaker_path)
        torch.save(self.listener.state_dict(), listener_path)

    def load_from(self, speaker_path, listener_path):
        self.speaker.load_state_dict(torch.load(speaker_path))
        self.listener.load_state_dict(torch.load(listener_path))
        self.old_speaker.load_state_dict(self.speaker.state_dict())
        self.old_listener.load_state_dict(self.listener.state_dict())


# ======================================
#  3. Define the Environment & Training
# ======================================

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

        if listener_action == self.current_obs.item():
            reward = 1
            done = True
        else:
            reward = -1
            done = done  # if we want to keep going or not

        return self.current_obs, reward, done, {}

# Create a MultiAgentPPO
obs_size = 2
language_size = 2
ts_limit = 1 # practically, max word length
multi_agent_ppo = MultiAgentPPO(obs_size, language_size, lr=1e-4)

PAD_TOKEN = -1
MAX_LEN = 10
    
def pad_sequence(seq) -> torch.LongTensor:
    """
    Pad a single sequence up to max_len with pad_token.
    seq: shape [seq_len]
    returns shape [max_len]
    """
    seq_len = seq.size(0)
    if seq_len < MAX_LEN:
        padding = torch.full((MAX_LEN - seq_len,), PAD_TOKEN, dtype=torch.long)
        return torch.cat([seq, padding], dim=0)
    else:
        return seq[:MAX_LEN]


def train(env, n_episodes=1000, mini_epochs=4):
    pbar = tqdm(range(n_episodes))
    loss_hist = []
    rewards_hist = []


    for episode in pbar:
        obs = env.reset()
        done = False

        # Collect transitions in memory
        # Each is: (obs, action, logprob, reward, mask, value)
        memory_speaker = ([], [], [], [], [], [])
        memory_listener = ([], [], [], [], [], [])

        total_reward = 0
        comms = []

        for _ in range(mini_epochs):
            while not done:
                # Speaker
                if len(comms) > 0:
                    speaker_obs = torch.cat([obs, torch.stack(comms)])
                else:
                    speaker_obs = obs

                speaker_action, speaker_log_prob = multi_agent_ppo.select_speaker_action(
                    speaker_obs, deterministic=True
                )
                comms.append(speaker_action)

                # Listener
                listener_obs = torch.stack(comms)
                listener_action, listener_log_prob = multi_agent_ppo.select_listener_action(
                    listener_obs, deterministic=True
                )
                listener_action_val = listener_action.item()

                # Step environment
                next_obs, reward, done, _ = env.step(
                    {"speaker": speaker_action, "listener": listener_action_val}
                )

                # Store Speaker memory
                memory_speaker[0].append(speaker_obs)
                memory_speaker[1].append(speaker_action)
                memory_speaker[2].append(speaker_log_prob.item())
                memory_speaker[3].append(reward)
                memory_speaker[4].append(0 if done else 1)
                # Value for speaker
                s_val = multi_agent_ppo.speaker(speaker_obs)[1].item()
                memory_speaker[5].append(s_val)

                # Store Listener memory
                memory_listener[0].append(listener_obs)
                memory_listener[1].append(listener_action_val)
                memory_listener[2].append(listener_log_prob.item())
                memory_listener[3].append(reward)
                memory_listener[4].append(0 if done else 1)
                # Value for listener
                l_val = multi_agent_ppo.listener(listener_obs)[1].item()
                memory_listener[5].append(l_val)

                obs = next_obs
                total_reward += reward

        # PPO update
        losses = multi_agent_ppo.train(memory_speaker, memory_listener, optimize=True)
        loss_hist.append(losses["loss_total"])
        rewards_hist.append(total_reward / mini_epochs)

        pbar.set_description(
            f"Episode: {episode+1}, Loss: {losses['loss_total']:.3f}, Reward: {total_reward/mini_epochs:.2f}"
        )

    return loss_hist, rewards_hist

def test(env, n_episodes=5):
    comm_map = "abcdefghijklmnopqrstuvwxyz"
    obs_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        comms = []
        comm_string = ""

        while not done:
            # Speaker
            speaker_action, _ = multi_agent_ppo.select_speaker_action(
                obs, deterministic=True
            )
            comm_string += comm_map[speaker_action]
            comms.append(speaker_action)

            # Listener
            listener_action, _ = multi_agent_ppo.select_listener_action(
                torch.stack(comms), deterministic=True
            )
            listener_action_val = listener_action.item()

            obs, reward, done, _ = env.step(
                {"speaker": speaker_action, "listener": listener_action_val}
            )
        
        target_char = obs_map[obs.item()]
        print(
            f"Episode: {episode+1}, Comm: {comm_string}, Target: {target_char}, Reward: {reward}"
        )

def find_translations(env, obs_size, n_episodes=10):

    comm_map = "abcdefghijklmnopqrstuvwxyz"
    obs_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    translations = {}

    for episode in range(n_episodes):
        for i in range(obs_size):
            # Reset environment with a specific observation (i)
            obs = env.reset(obj=i)
            done = False
            comms = []  # Store communications as they are generated
            comm_string = ""

            while not done:
                # Speaker generates a communication
                speaker_action, _ = multi_agent_ppo.select_speaker_action(obs, deterministic=True)
                comm_string += comm_map[speaker_action]
                comms.append(speaker_action)

                # Listener interprets the communication
                listener_action, _ = multi_agent_ppo.select_listener_action(torch.stack(comms), deterministic=True)

                # Perform the environment step with the speaker and listener's actions
                actions = {"speaker": speaker_action, "listener": listener_action.item()}
                obs, reward, done, _ = env.step(actions)

                print(f"For observation {obs}, speaker action: {speaker_action.item()}, "
                      f"listener action: {listener_action.item()}")

            # Map observations to translations
            if obs_map[i] not in translations:
                translations[obs_map[i]] = []

            # Only store successful translations (reward > 0)
            if reward > 0:
                translations[obs_map[i]].append(comm_string)

    return translations


if __name__ == "__main__":
    import os
    env = CommEnv(obs_size)

    env.ts_limit = ts_limit

    if not os.path.exists("speaker.pth") or not os.path.exists("listener.pth"):
        losses, avg_rewards = train(env, n_episodes=200, mini_epochs=1)

        import matplotlib.pyplot as plt
        plt.plot(losses, label="Total Loss (Speaker+Listener)")
        plt.legend()
        plt.show()
        plt.plot(avg_rewards, label="Average Rewards")
        plt.legend()
        plt.show()

        # Save
        multi_agent_ppo.save_to("speaker.pth", "listener.pth")
    else:
        multi_agent_ppo.load_from("speaker.pth", "listener.pth")

    # Test
    test(env, 5)
    translations = find_translations(env, obs_size, n_episodes=10)
    print(translations)
