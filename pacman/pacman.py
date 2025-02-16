import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import time
import random
import copy
import os

# ------------------------------
# Determine device (CPU, CUDA, MPS)
# ------------------------------
device = torch.device("mps")  # or "cuda" if you have a GPU
print(f"Using device: {device}")

# ------------------------------
# Q-Network CNN
# ------------------------------
class QNetworkCNN(nn.Module):
    """
    DQN-like CNN for an 84x84 grayscale input with 4-frame stacking.
    """
    def __init__(self, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        """
        Expects x of shape (B, 4, 84, 84) in [0, 255].
        """
        # Scale input from [0, 255] to [0, 1]
        if x.max() > 1.0:
            x = x / 255.0

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ------------------------------
# Hyperparameters
# ------------------------------
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99995
NUM_EPISODES = 550
MAX_EPISODE_LENGTH = 100000

BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.00025
TARGET_UPDATE_FREQUENCY = 1000  # Steps, not episodes
REPLAY_BUFFER_SIZE = 100000
MIN_REPLAY_BUFFER_SIZE = 1000

# ------------------------------
# Replay Buffer
# ------------------------------
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ------------------------------
# Epsilon-greedy Action Selection
# ------------------------------
def get_action(state, policy_net, epsilon, action_space):
    if np.random.rand() < epsilon:
        return action_space.sample()
    else:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = policy_net(state_t)
        return torch.argmax(q_values, dim=1).item()

# ------------------------------
# DQN Training Step
# ------------------------------
def train_step(policy_net, target_net, optimizer, replay_buffer, batch_size):
    batch = replay_buffer.sample(batch_size)
    if batch is None:
        return None

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # Q-values of current states
    q_values = policy_net(states)  # (B, action_size)
    q_values_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    # Target Q-values
    with torch.no_grad():
        next_q_values = target_net(next_states)  # (B, action_size)
        max_next_q_values = next_q_values.max(dim=1)[0]
        target_q_values = rewards + (1 - dones) * GAMMA * max_next_q_values

    loss = nn.MSELoss()(q_values_selected, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# ------------------------------
# Run a Single Episode
# ------------------------------
def run_episode(env, policy_net, target_net, optimizer, replay_buffer, epsilon):
    state, _ = env.reset()
    total_reward = 0.0
    total_loss = 0.0
    steps = 0

    for t in range(MAX_EPISODE_LENGTH):
        # Choose action
        action = get_action(state, policy_net, epsilon, env.action_space)
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        # Store in replay buffer
        replay_buffer.add((state, action, reward, next_state, float(done)))

        state = next_state
        total_reward += reward
        steps += 1

        # Training step
        if len(replay_buffer) >= MIN_REPLAY_BUFFER_SIZE:
            loss = train_step(policy_net, target_net, optimizer, replay_buffer, BATCH_SIZE)
            if loss is not None:
                total_loss += loss

        # Update target network periodically
        if steps % TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    avg_loss = total_loss / steps if steps > 0 else 0
    return total_reward, avg_loss, steps

# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':
    env_name = "ALE/MsPacman-v5"

    env = gym.make(env_name, frameskip=1)
    env = AtariPreprocessing(env)
    env = FrameStackObservation(env, stack_size=4)

    # Action size
    action_size = env.action_space.n
    env.close()

    # Tracking
    all_rewards_episodes = []
    all_losses_episodes = []
    avg_rewards_episodes = []
    max_rewards_episodes = []

    start_episode = 0
    policy_net = QNetworkCNN(action_size).to(device)
    target_net = QNetworkCNN(action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    avg_rewards_episodes = []
    max_rewards_episodes = []
    epsilon = EPSILON_START

    save_path = "pacman_dqn_checkpoint.pth"
    load_path = "pacman_dqn_checkpoint.pth"

    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        replay_buffer.buffer = deque(checkpoint['replay_buffer']) # Load replay buffer data
        all_rewards_episodes = checkpoint['all_rewards_episodes']
        all_losses_episodes = checkpoint['all_losses_episodes']
        avg_rewards_episodes = checkpoint['avg_rewards_episodes']
        max_rewards_episodes = checkpoint['max_rewards_episodes']
        epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode']
        print(f"Loaded checkpoint from episode {start_episode}")
    else:
        print("No checkpoint found, starting from scratch.")

    start_time = time.time()

    for episode in range(start_episode, NUM_EPISODES):
        print(f"--- Episode {episode + 1} ---")
        episode_start_time = time.time()

        env = gym.make(env_name, frameskip=1)
        env = AtariPreprocessing(env)
        env = FrameStackObservation(env, stack_size=4)

        reward, avg_loss, steps = run_episode(env, policy_net, target_net, optimizer, replay_buffer, epsilon)
        env.close()

        episode_end_time = time.time()
        all_rewards_episodes.append(reward)
        all_losses_episodes.append(avg_loss)

        # Compute stats
        avg_reward_ep = np.mean(all_rewards_episodes[-100:])  # avg over last 100
        max_reward_ep = np.max(all_rewards_episodes)
        avg_rewards_episodes.append(avg_reward_ep)
        max_rewards_episodes.append(max_reward_ep)

        print(f"  Epsilon: {epsilon:.4f}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Avg Reward (last 100): {avg_reward_ep:.2f}")
        print(f"  Max Reward So Far: {max_reward_ep:.2f}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Steps: {steps}")
        print(f"  Episode Time: {episode_end_time - episode_start_time:.2f} s")

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if (episode + 1) % 50 == 0:
            torch.save({
                'episode': episode + 1,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'replay_buffer': list(replay_buffer.buffer), # Save replay buffer data as list
                'all_rewards_episodes': all_rewards_episodes,
                'all_losses_episodes': all_losses_episodes,
                'avg_rewards_episodes': avg_rewards_episodes,
                'max_rewards_episodes': max_rewards_episodes,
                'epsilon': epsilon,
                }, save_path)
            print(f"Saved checkpoint at episode {episode + 1}")

        if (episode + 1) % 100 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(avg_rewards_episodes, label='Average Reward (last 100)')
            x_vals = np.arange(len(avg_rewards_episodes))
            y_vals = np.array(avg_rewards_episodes)
            coeffs = np.polyfit(x_vals, y_vals, 1)
            poly_fn = np.poly1d(coeffs)
            plt.plot(x_vals, poly_fn(x_vals), '--', color='red', label='Linear Fit')

    total_end_time = time.time()
    print(f"Total Training Time: {total_end_time - start_time:.2f} seconds")

    # Final Plot
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_episodes, label='Average Reward (last 100)')
    # plt.plot(max_rewards_episodes, label='Max Reward', linestyle='--')
    x_vals = np.arange(len(avg_rewards_episodes))
    y_vals = np.array(avg_rewards_episodes)
    coeffs = np.polyfit(x_vals, y_vals, 1)
    poly_fn = np.poly1d(coeffs)
    plt.plot(x_vals, poly_fn(x_vals), '--', color='red', label='Linear Fit')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN MsPacman Training Progress")
    plt.legend()
    plt.grid()
    plt.show()