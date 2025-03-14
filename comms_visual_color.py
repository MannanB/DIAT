import numpy as np
import random
import torch
from tqdm import tqdm
import os, pickle
import numpy as np
import matplotlib.pyplot as plt

from model import SpeakerListenerModel
from ppo import TwoActionPPO

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

create_agent = lambda: TwoActionPPO(SpeakerListenerModel, obs_size=OBS_SIZE, vocab_size=VOCAB_SIZE, action_size=len(shapes) + OBS_SIZE - 1, lr=LR, gamma=GAMMA, lam=LAMBDA, eps_clip=EPS_CLIP, entropy_coef=ENTROPY_COEF)

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
        agent.train(memory)
        reward_history.append(total_ep_reward)

        # balance ts limit increase in such a way that the last 10% of training is with max ts size
        # env.ts_limit = min(MAX_TURNS, int(MAX_TURNS * (ep / n_episodes) + 1))

        pbar.set_description(f"Ep {ep} Reward: {total_ep_reward:.2f}, Acc: {total_correct:.2f}")
        if ep % 1000 == 0:
            agent.save_to(f"checkpoint{ep + start_point}.pth")
            with open("rewards.pkl", "wb") as f:
                pickle.dump(reward_history, f)


    return reward_history, loss_history, speaker_probs_history

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
            agent = create_agent()
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

    agent = create_agent()
    for i in range(EPISODES, -1000, -1000):
        if os.path.exists(f"checkpoint{i}.pth"):
            agent.load_from(f"checkpoint{i}.pth")
            print(f"Loaded from checkpoint{i}.pth")
            break

    if not os.path.exists("speaker_listener.pth"):
        rewards, losses, dists = train(env, agent, n_episodes=EPISODES, start_point=i)
        agent.save_to("speaker_listener.pth")

        # Plot if desired
        import matplotlib.pyplot as plt
        import pickle
        # save losses
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