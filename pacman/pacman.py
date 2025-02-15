import gymnasium as gym
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
import multiprocessing
import os  # Import the 'os' module

# Determine device
device = torch.device("cpu")
print(f"Using device: {device}")

# --- Preprocessing and Frame Stacking ---
class PreprocessAndStack:
    def __init__(self, frame_stack_size=4):
        self.frame_stack_size = frame_stack_size
        self.frame_buffer = deque(maxlen=frame_stack_size)

    def _preprocess_single_frame(self, frame):
        """Grayscale and downsample a single frame."""
        frame = np.mean(frame, axis=2).astype(np.uint8)  # Grayscale
        frame = frame[::2, ::2]  # Downsample by a factor of 2
        return frame

    def reset(self, initial_frame):
        """Resets the buffer with the initial frame."""
        processed_frame = self._preprocess_single_frame(initial_frame)
        for _ in range(self.frame_stack_size):
            self.frame_buffer.append(processed_frame)
        return np.stack(self.frame_buffer, axis=0)  # (Stack_size, height, width)

    def step(self, frame):
        """Preprocesses and adds the new frame, returns stacked frames."""
        processed_frame = self._preprocess_single_frame(frame)
        self.frame_buffer.append(processed_frame)
        return np.stack(self.frame_buffer, axis=0)  # (Stack_size, height, width)

class QNetworkCNN(nn.Module):
    def __init__(self, action_size, frame_stack_size=4):
        super(QNetworkCNN, self).__init__()
        # Modified input channels to accept stacked frames
        self.conv1 = nn.Conv2d(in_channels=frame_stack_size, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        # Calculate the output size after the two conv layers (adjusted for downsampled size)
        self.fc1 = nn.Linear(32 * 11 * 8, 256)  # Adjusted for 105x80 input (after downsampling)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        """Expects x to be (B,frame_stack_size,105,80)"""
        # If coming in as (B, 105, 80, frame_stack_size)
        if x.ndimension() == 4 and x.shape[1] == 105 and x.shape[2] == 80:
            x = x.permute(0, 3, 1, 2)  # -> (B, frame_stack_size, 105, 80)

        # Convert from [0,255] or [0,1] to [0,1]
        if x.max() > 1.0:  # Assuming if max > 1, it's [0, 255]
            x = x / 255.0

        # Make sure x is contiguous before convolutions (often recommended)
        x = x.contiguous()

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Flatten all but the batch dimension
        x = x.flatten(start_dim=1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.9999
NUM_GENERATIONS = 1000
AGENTS_PER_GENERATION = 100
MAX_EPISODE_LENGTH = 10000
MUTATION_RATE = 0.01
MUTATION_STRENGTH = 1.0
TOP_AGENT_PERCENTAGE = 0.3
SAVE_INTERVAL = 100  # Save every 100 generations
SAVE_PATH = "pacman_training_data"  # Directory to save training data
FRAME_STACK_SIZE = 4  # Number of frames to stack
SKIP_FRAMES = 4  # Number of frames to skip


def get_action(state, agent, epsilon, action_space):
    """Choose action \epsilon-greedily from QNetwork."""
    if np.random.rand() < epsilon:
        return action_space.sample()
    else:
        # Convert state to torch tensor
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
        q_values = agent(state_t)
        return torch.argmax(q_values).item()

def evaluate_agent(agent, env, epsilon, preprocessor, render=False):
    """Evaluates a single agent and returns the total reward."""
    total_reward = 0
    state, _ = env.reset()
    state = preprocessor.reset(state)  # Reset preprocessor
    done = False

    for _ in range(100000000):  # Large upper bound
        action = get_action(state, agent, epsilon, env.action_space)

        accumulated_reward = 0
        for _ in range(SKIP_FRAMES):
            next_state, reward, terminated, truncated, info = env.step(action)
            accumulated_reward += reward
            if terminated or truncated:
                break
        done = terminated or truncated
        next_state = preprocessor.step(next_state)  # Preprocess and stack

        total_reward += accumulated_reward
        state = next_state

        if render:
            env.render()

        if done:
            break

    return total_reward


def run_agent_episode(agent_index, agent, env_name, epsilon, generation_number):
    env = gym.make(env_name, render_mode="rgb_array")
    preprocessor = PreprocessAndStack(frame_stack_size=FRAME_STACK_SIZE)

    reward = evaluate_agent(agent, env, epsilon, preprocessor)
    env.close()
    return reward


def run_generation(agents, env_name, epsilon, generation_number):
    """Runs all agents in a generation in parallel and evaluates them."""
    num_agents = len(agents)

    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # Prepare arguments for each worker process
        worker_args = [
            (i, agents[i], env_name, epsilon, generation_number)
            for i in range(num_agents)
        ]

        # Run agent episodes in parallel
        rewards = pool.starmap(run_agent_episode, worker_args)

    return rewards


def mutate(agent):
    """Mutates the weights of a neural network agent."""
    for param in agent.parameters():
        if np.random.rand() < MUTATION_RATE:
            mutation = torch.randn_like(param) * MUTATION_STRENGTH
            param.data.add_(mutation)

def evolve_population(agents, rewards):
    """Evolves the population using selection and mutation."""
    num_agents = len(agents)
    num_top_agents = int(TOP_AGENT_PERCENTAGE * num_agents)

    # Sort agents by reward (descending)
    agent_reward_pairs = sorted(zip(agents, rewards), key=lambda x: x[1], reverse=True)
    sorted_agents = [agent for agent, reward in agent_reward_pairs]

    # Keep the top agents
    top_agents = sorted_agents[:num_top_agents]

    new_agents = []
    for agent in top_agents:
        new_agents.append(copy.deepcopy(agent))

    # Fill the rest of the population by mutating random top agents
    while len(new_agents) < num_agents:
        parent_agent = random.choice(top_agents)
        child_agent = copy.deepcopy(parent_agent)
        mutate(child_agent)
        new_agents.append(child_agent)

    return new_agents

def save_training_data(generation, agents, avg_rewards, max_rewards, epsilon, save_path):
    """Saves the training data (agents, rewards, and epsilon)."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint_path = os.path.join(save_path, f"generation_{generation}.pth")
    agent_states = [agent.state_dict() for agent in agents]

    torch.save({
        'generation': generation,
        'agents': agent_states,
        'avg_rewards': avg_rewards,
        'max_rewards': max_rewards,
        'epsilon': epsilon,
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_training_data(save_path, action_size, frame_stack_size):
    """Loads the training data and returns the agents, rewards, and epsilon."""
    checkpoint_files = [f for f in os.listdir(save_path) if f.endswith('.pth')]
    if not checkpoint_files:
        print("No checkpoint files found. Starting from scratch.")
        return 0, [QNetworkCNN(action_size, frame_stack_size).to(device) for _ in range(AGENTS_PER_GENERATION)], [], [], EPSILON_START

    # Find the latest generation checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[1].split('.')[0]))
    checkpoint_path = os.path.join(save_path, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)

    generation = checkpoint['generation']
    agent_states = checkpoint['agents']
    avg_rewards = checkpoint['avg_rewards']
    max_rewards = checkpoint['max_rewards']
    epsilon = checkpoint['epsilon']

    agents = [QNetworkCNN(action_size, frame_stack_size).to(device) for _ in range(len(agent_states))]
    for i, state_dict in enumerate(agent_states):
        agents[i].load_state_dict(state_dict)

    print(f"Loaded checkpoint from {checkpoint_path} (Generation {generation})")
    return generation, agents, avg_rewards, max_rewards, epsilon


if __name__ == '__main__':
    env_name = "ALE/MsPacman-v5"

    test_env = gym.make(env_name)
    action_size = test_env.action_space.n
    test_env.close()

    # Track statistics
    all_rewards_generations = []
    avg_rewards_generations = []
    max_rewards_generations = []

    # Load or initialize agents
    start_generation, agents, avg_rewards_generations, max_rewards_generations, epsilon = load_training_data(SAVE_PATH, action_size, FRAME_STACK_SIZE)
    if start_generation >0:
      all_rewards_generations = [[0]*AGENTS_PER_GENERATION]* start_generation #Just to make code below compatible
    start_time = time.time()  # Total start time

    for generation in range(start_generation, NUM_GENERATIONS):
        print(f"--- Generation {generation + 1} ---")

        generation_start_time = time.time()

        # Evaluate agents in parallel
        rewards = run_generation(agents, env_name, epsilon, generation)

        generation_end_time = time.time()

        all_rewards_generations.append(rewards)  # Append *after* evaluation
        avg_reward_gen = sum(rewards) / AGENTS_PER_GENERATION
        max_reward_gen = max(rewards)
        avg_rewards_generations.append(avg_reward_gen)
        max_rewards_generations.append(max_reward_gen)


        print(f"  Epsilon: {epsilon:.4f}")
        print("  Rewards for each agent:", rewards)
        print(f"  Average Reward: {avg_reward_gen:.2f}")
        print(f"  Max Reward: {max_reward_gen:.2f}")
        print(f"  Generation Time: {generation_end_time - generation_start_time:.2f} seconds")

        # Evolve the population
        agents = evolve_population(agents, rewards)

        # Decay epsilon
        epsilon = EPSILON_START * (EPSILON_END / EPSILON_START) ** ((generation + 1) / NUM_GENERATIONS) # +1 to match save
        epsilon = max(epsilon, EPSILON_END)


        # Save training data
        if (generation + 1) % SAVE_INTERVAL == 0:
            save_training_data(generation + 1, agents, avg_rewards_generations, max_rewards_generations, epsilon, SAVE_PATH)

        # Plot every 100 generations
        if (generation + 1) % 100 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(avg_rewards_generations, label='Average Reward')

            # Linear regression line for average rewards
            x = np.arange(len(avg_rewards_generations))
            y = np.array(avg_rewards_generations)
            coeffs = np.polyfit(x, y, 1)
            linear_fit = np.poly1d(coeffs)
            plt.plot(x, linear_fit(x), label='Linear Fit', linestyle='--', color='red')

            plt.xlabel("Generation")
            plt.ylabel("Reward")
            plt.title("Evolutionary Pacman Training Progress")
            plt.legend()
            plt.grid(True)
            plt.show()

    total_end_time = time.time()
    print(f"Total Training Time: {total_end_time - start_time:.2f} seconds")

    # Final Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_generations, label='Average Reward')
    plt.plot(max_rewards_generations, label='Max Reward', linestyle='--')

    x = np.arange(len(avg_rewards_generations))
    y = np.array(avg_rewards_generations)
    coeffs = np.polyfit(x, y, 1)
    linear_fit = np.poly1d(coeffs)
    plt.plot(x, linear_fit(x), label='Linear Fit', linestyle='--', color='red')

    plt.xlabel("Generation")
    plt.ylabel("Reward")
    plt.title("Evolutionary Pacman Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()