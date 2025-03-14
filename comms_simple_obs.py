import numpy as np
import random
import torch
from tqdm import tqdm
import os, pickle
import matplotlib.pyplot as plt

from model import SpeakerListenerModel
from ppo import PPO

MAX_SEQ_LEN = 10  # Maximum length of communication sequence
OBS_SIZE = 10     # Number of possible observations
VOCAB_SIZE = 3    # Size of "language" for the speaker
MAX_TURNS = 4     # Max # of turns (also limits seq lenght)
EPISODES = 10000
MINI_EPISODES = 4
LR = 1e-4
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.1
ENTROPY_COEF = 0.05


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

        if listener_action == self.current_obs.item():
            reward = (self.ts_limit + 1) / 2
            done = True
        else:
            reward = -(self.timesteps / self.ts_limit)
        
        return self.current_obs, reward, done, {}

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

        for _ in range(n_mini_episodes):
            _, obs = env.reset()
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
        agent.train(memory)
        reward_history.append(ep_reward)

        pbar.set_description(f"Ep {ep} Reward: {ep_reward:.2f}")
        if ep % 1000 == 0:
            agent.save_to(f"checkpoint{ep}.pth")
            with open("rewards.pkl", "wb") as f:
                pickle.dump(reward_history, f)


    return reward_history, loss_history, speaker_probs_history

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
    total_correct = 0
    total_translations = 0
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
                total_correct += 1
            total_translations += 1
    return translations, total_correct / total_translations

def create_accuracy_graph(env, folder_path):
    acc = []
    episodes = []
    translations = []
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith(".pth") and "listener" not in file and "Best" not in file:
            agent = PPO(SpeakerListenerModel, obs_size=OBS_SIZE, vocab_size=VOCAB_SIZE, action_size=OBS_SIZE, lr=LR, gamma=GAMMA, lmbda=LAMBDA, eps_clip=EPS_CLIP, entropy_coef=ENTROPY_COEF)
            agent.load_from(os.path.join(folder_path, file))
            tl, acci = get_translations(env, agent, n_episodes=4)
            acc.append(acci)
            translations.append(tl)
            episodes.append(int(file.split(".")[0][10:]))
    acc_sorted = sorted(zip(episodes, acc, translations), key=lambda x: x[0])
    episodes, acc, translations = zip(*acc_sorted)
    plt.plot(episodes, acc)
    plt.title("Accuracy")
    plt.xlabel("Episodes")
    plt.ylabel("Accuracy")
    plt.show()
    return translations

def TNSEProjection(agent):
    # project vocab to 2D using TSNE for only the listener embeddings

    # get all the embeddings
    embeddings = []
    for i in range(VOCAB_SIZE):
        comm_tokens = torch.tensor([i]).view(1, 1)
        embedding = agent.policy.listener_net.transformer.embedding(comm_tokens)
        embeddings.append(embedding.squeeze(0).detach().numpy())

    embeddings = np.array(embeddings).squeeze(1)
    print(embeddings.shape)

    # project to 2D
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0, perplexity=1)
    embeddings_2d = tsne.fit_transform(embeddings)
    print(embeddings_2d.shape)

    # plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    vocab_map = "abcdefghijklmnopqrstuvwxyz"
    for i, txt in enumerate(range(VOCAB_SIZE)):
        plt.annotate(vocab_map[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    plt.show()

    return embeddings_2d
# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    env = CommEnv(OBS_SIZE, ts_limit=MAX_TURNS)

    agent = PPO(SpeakerListenerModel, obs_size=OBS_SIZE, vocab_size=VOCAB_SIZE, action_size=OBS_SIZE, lr=LR, gamma=GAMMA, lmbda=LAMBDA, eps_clip=EPS_CLIP, entropy_coef=ENTROPY_COEF)

    for i in range(EPISODES, -1000, -1000):
        if os.path.exists(f"checkpoint{i}.pth"):
            agent.load_from(f"checkpoint{i}.pth")
            print(f"Loaded from checkpoint{i}.pth")
            break

    if not os.path.exists("speaker_listener.pth"):
        rewards, losses, dists = train(env, agent, n_episodes=EPISODES, start_point=i)
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


        speaker_probs_history = np.array(dists)  # shape [T, vocab_size]
        for token_id in range(VOCAB_SIZE):
            plt.plot(speaker_probs_history[:, token_id], label=f"Token {token_id}")
        plt.legend()
        plt.show()
    
    test(env, agent, n_episodes=5)
    ts, _ = get_translations(env, agent, n_episodes=10)
    print(ts)