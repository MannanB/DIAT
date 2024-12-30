import torch
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(self, model, obs_size, vocab_size, gamma=.99, eps_clip=.2, lr=1e-4, lmbda=0.95, entropy_coef=0.1):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef

        self.policy = model(obs_size, vocab_size)
        self.old_policy = model(obs_size, vocab_size)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-4)

    def convert_logits_to_action_and_logprob(self, logits, actions):
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprobs = dist.log_prob(actions)
        return action, logprobs

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
        speaker_action, speaker_logprob = self.convert_logits_to_action_and_logprob(speaker_logits, speaker_action) 

        # 2) Append speaker_action to the communication tokens
        updated_comm_tokens = torch.cat(
            [comm_tokens, speaker_action.unsqueeze(-1)], dim=1
        )

        # 3) Listener forward
        listener_logits, listener_value = self.old_policy.forward_listener(updated_comm_tokens)
        listener_action, listener_logprob = self.convert_logits_to_action_and_logprob(listener_logits, listener_action)

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
    
    def calculate_surrogate(self, ratio, advantages):
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        return -torch.min(surr1, surr2)
    
    def calculate_critics_loss(self, old_val, new_val, ret):
        value_pred_listener_clipped = old_val + (ret - old_val).clamp(
            -self.eps_clip, self.eps_clip
        )
        listener_val_loss_clipped = (value_pred_listener_clipped - ret) ** 2
        listener_val_loss_normal = (new_val - ret) ** 2
        return 0.5 * torch.max(listener_val_loss_clipped, listener_val_loss_normal)

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

                speaker_logits, speaker_val = self.policy.forward_speaker(obs_t, pad_mask=None)
                speaker_probs = torch.softmax(speaker_logits, dim=-1)
                speaker_dist = torch.distributions.Categorical(speaker_probs)
                new_speaker_logprob = speaker_dist.log_prob(speaker_actions[t].unsqueeze(0))
                speaker_entropy = speaker_dist.entropy().mean()

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

                actor_loss_speaker = self.calculate_surrogate(ratio_speaker, adv_speaker)
                actor_loss_listener = self.calculate_surrogate(ratio_listener, adv_listener)

                # Critic losses
                ret_speaker = returns_speaker[t]
                ret_listener = returns_listener[t]

                critic_loss_speaker = self.calculate_critics_loss(speaker_values[t], speaker_val, ret_speaker)
                critic_loss_listener = self.calculate_critics_loss(listener_values[t], listener_val, ret_listener)

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