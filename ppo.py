import torch
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(self, model, obs_size, vocab_size, action_size, gamma=.99, eps_clip=.2, lr=1e-4, lmbda=0.95, entropy_coef=0.1, listener_obs_size=None):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef

        self.policy = model(obs_size, vocab_size, action_size, listener_obs_size=listener_obs_size)
        self.old_policy = model(obs_size, vocab_size, action_size, listener_obs_size=listener_obs_size)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-4)

    def process_logits(self, logits, action=None):
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = dist.sample()
        logprobs = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, logprobs, entropy
    
    def forward_speaker(self, obs_tokens):
        # get action, logprob, entropy
        logits, value = self.old_policy.forward_speaker(obs_tokens)
        action, logprob, entropy = self.process_logits(logits)
        return action, logprob, value, entropy

    def forward_listener(self, comm_tokens):
        logits, value = self.old_policy.forward_listener(comm_tokens)
        action, logprob, entropy = self.process_logits(logits)
        return action, logprob, value, entropy

    def select_action(self, obs_tokens, comm_tokens, encoding_offset = 0):
        """
        obs_tokens: [batch_size, seq_len_obs], but we have 1 environment => shape (1, seq_len_obs)
        comm_tokens: [batch_size, seq_len_comm]
        
        We'll produce:
          speaker_action: next token from speaker
          listener_action: guessed obs from listener
        """
        # 1) Speaker forward
        speaker_logits, speaker_value = self.old_policy.forward_speaker(obs_tokens)
        speaker_action, speaker_logprob, _ = self.process_logits(speaker_logits)

        # 2) Append speaker_action to the communication tokens
        updated_comm_tokens = torch.cat(
            [comm_tokens, speaker_action.unsqueeze(-1) + encoding_offset], dim=1
        )

        # 3) Listener forward
        listener_logits, listener_value = self.old_policy.forward_listener(updated_comm_tokens)
        listener_action, listener_logprob, _ = self.process_logits(listener_logits)

        return (speaker_action, speaker_logprob, speaker_value,
                listener_action, listener_logprob, listener_value)

    def compute_gae(self, rewards, masks, values, next_value = 0):
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
    
    def calculate_surrogate(self, new_logprob, old_logprob, advantages):
        ratio = torch.exp(new_logprob - old_logprob)
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
        speaker_actions = torch.LongTensor(memory['speaker_actions'])
        speaker_logprobs = torch.FloatTensor(memory['speaker_logprobs'])
        speaker_values = torch.FloatTensor(memory['speaker_values'])
        listener_actions = torch.LongTensor(memory['listener_actions'])
        listener_logprobs = torch.FloatTensor(memory['listener_logprobs'])
        listener_values = torch.FloatTensor(memory['listener_values'])
        rewards = np.array(memory['rewards'])
        listener_masks = np.array(memory['listener_masks'])
        speaker_masks = np.array(memory['speaker_masks'])

        if rewards.std() > 1e-8:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards = rewards - rewards.mean()

        # Next value estimates (last step)
        next_value_speaker = 0.0
        next_value_listener = 0.0

        # Compute advantages & returns
        advantages_speaker, returns_speaker = self.compute_gae(rewards, speaker_masks, speaker_values, next_value_speaker)
        advantages_listener, returns_listener = self.compute_gae(rewards, listener_masks, listener_values, next_value_listener)

        # For multiple minibatch updates
        for _ in range(4):
            # We'll accumulate losses across all steps, then do one optimizer step.
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
                _, new_speaker_logprob, speaker_entropy = self.process_logits(speaker_logits, speaker_actions[t])

                updated_comm = torch.cat([comm_t, speaker_actions[t].view(1,1)], dim=1)

                listener_logits, listener_val = self.policy.forward_listener(updated_comm, pad_mask=None)
                _, new_listener_logprob, listener_entropy = self.process_logits(listener_logits, listener_actions[t])

                adv_speaker = advantages_speaker[t]
                adv_listener = advantages_listener[t]

                actor_loss_speaker = self.calculate_surrogate(new_speaker_logprob, speaker_logprobs[t], adv_speaker)
                actor_loss_listener = self.calculate_surrogate(new_listener_logprob, listener_logprobs[t], adv_listener)

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


class TwoActionPPO(PPO):
    def __init__(self, model, obs_size, vocab_size, action_size, gamma=.99, eps_clip=.2, lr=1e-4, lmbda=0.95, entropy_coef=0.1):
        super(TwoActionPPO, self).__init__(model, obs_size, vocab_size, action_size, gamma, eps_clip, lr, lmbda, entropy_coef)

    def select_action(self, obs_shape, comm_tokens):
        # 1) Speaker forward
        speaker_logits, speaker_value = self.old_policy.forward_speaker(obs_shape)
        speaker_action, speaker_logprob, _ = self.process_logits(speaker_logits)

        # 2) Append speaker_action to the communication tokens
        updated_comm_tokens = torch.cat(
            [comm_tokens, speaker_action.unsqueeze(-1)], dim=1
        )

        # 3) Listener forward
        listener_logits_shape, listener_logits_color, listener_value = self.old_policy.forward_listener(updated_comm_tokens)

        listener_action_shape, listener_logprob_shape, _ = self.process_logits(listener_logits_shape)
        listener_action_color, listener_logprob_color, _ = self.process_logits(listener_logits_color)

        return (speaker_action, speaker_logprob, speaker_value,
                (listener_action_shape, listener_action_color),
                (listener_logprob_shape, listener_logprob_color), listener_value)

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
        speaker_actions = torch.LongTensor(memory['speaker_actions'])
        speaker_logprobs = torch.FloatTensor(memory['speaker_logprobs'])
        speaker_values = torch.FloatTensor(memory['speaker_values'])
        listener_actions = torch.LongTensor(memory['listener_actions'])
        listener_logprobs = torch.FloatTensor(memory['listener_logprobs'])
        listener_values = torch.FloatTensor(memory['listener_values'])
        rewards = np.array(memory['rewards'])
        masks = np.array(memory['masks'])

        listener_logprobs_shape = listener_logprobs[:, 0]
        listener_logprobs_color = listener_logprobs[:, 1]
        listener_action_shape = listener_actions[:, 0]
        listener_action_color = listener_actions[:, 1]

        # Next value estimates (last step)
        next_value_speaker = 0.0
        next_value_listener = 0.0

        # Compute advantages & returns
        advantages_speaker, returns_speaker = self.compute_gae(rewards, masks, speaker_values, next_value_speaker)
        advantages_listener, returns_listener = self.compute_gae(rewards, masks, listener_values, next_value_listener)

        # For multiple minibatch updates
        for _ in range(4):

            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_entropy = 0.0

            T = len(obs_tokens)
            for t in range(T):
                obs_t = obs_tokens[t]     # shape [1, obs_seq_len]
                comm_t = comm_tokens[t]   # shape [1, comm_seq_len]

                obs_t = torch.LongTensor(obs_t).unsqueeze(0)   # [1, obs_seq_len]
                comm_t = torch.LongTensor(comm_t).unsqueeze(0) # [1, comm_seq_len]

                # speaker forward
                speaker_logits, speaker_val = self.policy.forward_speaker(obs_t, pad_mask=None)
                _, new_speaker_logprob, speaker_entropy = self.process_logits(speaker_logits, speaker_actions[t])

                # listener forward
                updated_comm = torch.cat([comm_t, speaker_actions[t].view(1,1)], dim=1)
                listener_logits_shape, listener_logits_color, listener_val = self.policy.forward_listener(updated_comm, pad_mask=None)
                _, new_listener_logprob_shape, listener_entropy_shape = self.process_logits(listener_logits_shape, listener_action_shape[t])
                _, new_listener_logprob_color, listener_entropy_color = self.process_logits(listener_logits_color, listener_action_color[t])
                # Sum log probabilities (for combined optimization later)
                listener_entropy = (listener_entropy_shape + listener_entropy_color) / 2

                actor_loss_speaker = self.calculate_surrogate(new_speaker_logprob, speaker_logprobs[t], advantages_speaker[t])

                actor_loss_listener_shape = self.calculate_surrogate(new_listener_logprob_shape, listener_logprobs_shape[t], advantages_listener[t])
                actor_loss_listener_color = self.calculate_surrogate(new_listener_logprob_color, listener_logprobs_color[t], advantages_listener[t])
                actor_loss_listener = actor_loss_listener_shape + actor_loss_listener_color
                # Critic losses
                critic_loss_speaker = self.calculate_critics_loss(speaker_values[t], speaker_val, returns_speaker[t])
                critic_loss_listener = self.calculate_critics_loss(listener_values[t], listener_val, returns_listener[t])
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

class PPOEntireCommunicationFirst:
    def __init__(self, model, obs_size, vocab_size, action_size, n_comm_string,
                 gamma=0.99, eps_clip=0.2, lr=1e-4, lmbda=0.95, entropy_coef=0.1,
                 listener_obs_size=None):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef
        self.n_comm_string = n_comm_string

        # Initialize policy and old_policy.
        self.policy = model(obs_size, vocab_size, action_size, listener_obs_size=listener_obs_size)
        self.old_policy = model(obs_size, vocab_size, action_size, listener_obs_size=listener_obs_size)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-4)

    def process_logits(self, logits, action=None, masked_action=None):
        if masked_action is not None:
            logits[0][masked_action] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = dist.sample()
        logprobs = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, logprobs, entropy
    
    def create_comms(self, obs_tokens, init_comm_tokens=None, encoding_offset=0):
        """
        Autoregressively generates an entire communication string.
        Each generation step concatenates the tokens generated so far (if any) to the obs_tokens.
        """
        if init_comm_tokens is None:
            init_comm_tokens = torch.empty((obs_tokens.size(0), 0), dtype=torch.long, device=obs_tokens.device)
        generated_comm = init_comm_tokens
        speaker_actions = []
        speaker_logprobs = []
        speaker_values = []
        for _ in range(self.n_comm_string):
            # Autoregressive input: observation tokens concatenated with previously generated tokens.
            speaker_input = torch.cat([obs_tokens, generated_comm], dim=-1)
            speaker_logits, speaker_value = self.old_policy.forward_speaker(speaker_input)
            action, logprob, _ = self.process_logits(speaker_logits)
            speaker_actions.append(action)
            speaker_logprobs.append(logprob)
            speaker_values.append(speaker_value)
            # Append the new token to the generated communication.
            generated_comm = torch.cat([generated_comm, action.unsqueeze(-1)], dim=1)

        # # find index of 2 in obs_tokens
        # two_idx = torch.where(obs_tokens == 2 + 3)[1]
        # combs = []
        # for n in range(3):
        #     for m in range(3):
        #         for l in range(3):
        #             combs.append([n, m, l])
        
        # generated_comm2 = combs[two_idx.item()]


        return (speaker_actions, speaker_logprobs, speaker_values, generated_comm)
    
    def select_action_based_comms(self, generated_comm, encoding_offset=0, listener_obs=None, masked_action=None):
        if listener_obs is not None:
            full_comm = torch.cat([listener_obs, generated_comm + encoding_offset], dim=1)
        else:
            full_comm = generated_comm
        # Now, pass the full communication string to the listener.
        listener_logits, listener_value = self.old_policy.forward_listener(full_comm)
        listener_action, listener_logprob, _ = self.process_logits(listener_logits, masked_action=masked_action)
        return (listener_action, listener_logprob, listener_value)

    def compute_gae(self, rewards, masks, values, next_value=0):
        """
        Computes the Generalized Advantage Estimator (GAE).
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

    def calculate_surrogate(self, new_logprob, old_logprob, advantages):
        ratio = torch.exp(new_logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        return -torch.min(surr1, surr2)

    def calculate_critics_loss(self, old_val, new_val, ret):
        # Clipped value loss.
        value_pred_clipped = old_val + (ret - old_val).clamp(-self.eps_clip, self.eps_clip)
        loss_clipped = (value_pred_clipped - ret) ** 2
        loss_unclipped = (new_val - ret) ** 2
        return 0.5 * torch.max(loss_clipped, loss_unclipped)

    def train(self, memory, train_listener=True, train_speaker=True):
        """
        Memory is a dictionary containing (per round):
          - obs_tokens: list of observation tokens (already shifted), e.g. [seq_len_obs]
          - comm_tokens: list of full communication tokens generated by the speaker [n_comm_string]
          - speaker_actions: list (length n_comm_string) of tokens produced by the speaker
          - speaker_logprobs: list (length n_comm_string) of log probabilities for each speaker token
          - speaker_values: list (length n_comm_string) of value estimates for each speaker token
          - listener_actions: scalar action from the listener
          - listener_logprobs: listener log probability (scalar)
          - listener_values: listener value estimate (scalar)
          - rewards: scalar reward for the round (applied to both speaker and listener)
          - speaker_masks and listener_masks: masks (1 if non-terminal)
        """
        T = len(memory['obs_tokens'])
        # Convert stored lists to tensors.
        speaker_actions = torch.LongTensor(memory['speaker_actions'])      # [T, n_comm_string]
        speaker_logprobs = torch.FloatTensor(memory['speaker_logprobs'])     # [T, n_comm_string]
        speaker_values = torch.FloatTensor(memory['speaker_values'])         # [T, n_comm_string]
        listener_actions = torch.LongTensor(memory['listener_actions'])      # [T]
        listener_logprobs = torch.FloatTensor(memory['listener_logprobs'])     # [T]
        listener_values = torch.FloatTensor(memory['listener_values'])         # [T]
        rewards = np.array(memory['rewards'])
        speaker_masks = np.array(memory['speaker_masks'])
        listener_masks = np.array(memory['listener_masks'])

        # Normalize rewards.
        if rewards.std() > 1e-8:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards = rewards - rewards.mean()

        # Compute advantages and returns for the listener.
        adv_listener, ret_listener = self.compute_gae(rewards, listener_masks, listener_values, next_value=0)

        # For the speaker, compute advantages per token in the communication string.
        speaker_advantages = []
        speaker_returns = []
        for t in range(T):
            # Use the same round reward for every token.
            rewards_seq = [rewards[t]] * self.n_comm_string
            masks_seq = [1.0] * self.n_comm_string
            values_seq = [speaker_values[t][-1]] * self.n_comm_string
            adv_seq, ret_seq = self.compute_gae(rewards_seq, masks_seq, values_seq, next_value=0)
            # Average the token-wise advantages/returns to get a round-level signal.
            speaker_advantages.append(torch.mean(adv_seq))
            speaker_returns.append(torch.mean(ret_seq))

        speaker_advantages = torch.stack(speaker_advantages)
        speaker_returns = torch.stack(speaker_returns)

        # Perform multiple epochs over the collected data.
        for _ in range(4):
            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_entropy = 0.0

            for t in range(T):
                # Recreate the observation for round t.
                obs_t_list = memory['obs_tokens'][t]  # e.g., a list of token IDs
                obs_t = torch.LongTensor(obs_t_list).unsqueeze(0)  # shape [1, seq_len_obs]

                # --- Speaker loss (autoregressive) ---
                actor_loss_speaker = 0.0
                speaker_entropy_sum = 0.0
                new_speaker_values = []
                # We'll re-generate the communication token by token.
                generated_comm = torch.empty((1, 0), dtype=torch.long)
                for j in range(self.n_comm_string):
                    # Input is observation tokens concatenated with previously generated tokens.
                    speaker_input = torch.cat([obs_t, generated_comm], dim=-1)
                    speaker_logits, new_speaker_value = self.policy.forward_speaker(speaker_input)
                    # Use stored speaker action for token j.
                    token = torch.tensor(memory['speaker_actions'][t][j]).unsqueeze(0)
                    _, new_logprob, token_entropy = self.process_logits(speaker_logits, token)
                    actor_loss_speaker += self.calculate_surrogate(new_logprob,
                                                                    speaker_logprobs[t][j],
                                                                    speaker_advantages[t])
                    speaker_entropy_sum += token_entropy
                    new_speaker_values.append(new_speaker_value)
                    # Append the stored token to simulate autoregressive generation.
                    generated_comm = torch.cat([generated_comm, token.unsqueeze(-1)], dim=1)
                actor_loss_speaker /= self.n_comm_string
                speaker_entropy_avg = speaker_entropy_sum / self.n_comm_string

                # Critic loss for speaker: compare the average new value to the stored average.
                new_speaker_avg_val = torch.mean(torch.stack(new_speaker_values))
                old_speaker_avg_val = torch.mean(speaker_values[t])
                critic_loss_speaker = self.calculate_critics_loss(old_speaker_avg_val,
                                                                  new_speaker_avg_val,
                                                                  speaker_returns[t])

                # --- Listener loss ---
                comm_t_list = memory['comm_tokens'][t]  # full communication tokens as stored.
                comm_t = torch.LongTensor(comm_t_list).unsqueeze(0)  # shape [1, n_comm_string]
                listener_logits, new_listener_value = self.policy.forward_listener(comm_t)
                listener_action_token = torch.tensor(memory['listener_actions'][t]).unsqueeze(0)
                _, new_listener_logprob, listener_entropy = self.process_logits(listener_logits,
                                                                                 listener_action_token)
                actor_loss_listener = self.calculate_surrogate(new_listener_logprob,
                                                               listener_logprobs[t],
                                                               adv_listener[t])
                critic_loss_listener = self.calculate_critics_loss(listener_values[t],
                                                                   new_listener_value,
                                                                   ret_listener[t])
                if train_listener:
                    total_actor_loss += actor_loss_listener
                    total_critic_loss += critic_loss_listener
                    total_entropy += listener_entropy
                if train_speaker:
                    total_actor_loss += actor_loss_speaker
                    total_critic_loss += critic_loss_speaker
                    total_entropy += speaker_entropy_avg

            # Average over the T rounds.
            total_actor_loss /= T
            total_critic_loss /= T
            total_entropy /= T

            loss = total_actor_loss + total_critic_loss - self.entropy_coef * total_entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update the old_policy with the new policy weights.
        self.old_policy.load_state_dict(self.policy.state_dict())
        return loss.item()

    def save_to(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_from(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.old_policy.load_state_dict(self.policy.state_dict())
