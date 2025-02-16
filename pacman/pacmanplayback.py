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
# Epsilon-greedy Action Selection
# ------------------------------
def get_action(state, policy_net, epsilon, action_space):
    if np.random.rand() < epsilon:
        return action_space.sample()
    else:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = policy_net(state_t)
        return torch.argmax(q_values, dim=1).item()

if __name__ == '__main__':
    env_name = "ALE/MsPacman-v5"
    load_path = "pacman_dqn_checkpoint.pth"

    env = gym.make(env_name, render_mode="human", frameskip=1) # Set render_mode to "human"
    env = AtariPreprocessing(env)
    env = FrameStackObservation(env, stack_size=4)

    action_size = env.action_space.n

    policy_net = QNetworkCNN(action_size).to(device)

    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        print(f"Loaded checkpoint from {load_path}")
    else:
        print(f"Error: Checkpoint file not found at {load_path}")
        exit()

    state, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not terminated and not truncated:
        action = get_action(state, policy_net, 0.0, env.action_space) # Exploitation, epsilon=0
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        time.sleep(0.02) # Slow down rendering for better visualization

    print(f"Total reward: {total_reward}")
    env.close()