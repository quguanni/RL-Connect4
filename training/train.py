import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from environment.connect4_env import Connect4
from models.alphazero_net import AlphaZeroNet
from mcts.mcts import run_mcts
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        # Initialize a replay buffer with a fixed maximum capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value):
        # Add a new experience (state, policy, value) to the buffer
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        # Randomly sample a batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        # Convert the sampled data into PyTorch tensors
        return torch.tensor(np.array(states), dtype=torch.float32), \
               torch.tensor(np.array(policies), dtype=torch.float32), \
               torch.tensor(np.array(values), dtype=torch.float32)

def self_play_episode(env, net, simulations=50):
    # Simulate a single self-play episode
    history = []  # Store the state and policy history
    env.reset()  # Reset the environment to the initial state
    done = False  # Track whether the game is over

    while not done:
        state = env.get_state()  # Get the current state of the environment
        pi = run_mcts(env.clone(), net, simulations)  # Run MCTS to get action probabilities
        action = np.random.choice(7, p=pi)  # Sample an action based on the probabilities
        history.append((state, pi))  # Save the state and policy
        _, reward, done = env.step(action)  # Take the action and observe the result

    # Return the history with rewards adjusted for alternating players
    return [(s, p, reward if i % 2 == 0 else -reward) for i, (s, p) in enumerate(history)]

def train():
    # Train the AlphaZero network
    net = AlphaZeroNet()  # Initialize the neural network

    # Load from checkpoint if it exists
    if os.path.exists("checkpoint_1000.pt"):
        print("Loading checkpoint_1000.pt...")
        net.load_state_dict(torch.load("checkpoint_1000.pt"))  # Load the saved model weights

    optimizer = optim.Adam(net.parameters(), lr=1e-3)  # Set up the optimizer
    buffer = ReplayBuffer(capacity=10000)  # Initialize the replay buffer

    for episode in range(1001, 20001):  # Loop through training episodes
        env = Connect4()  # Create a new Connect4 environment
        episode_data = self_play_episode(env, net)  # Generate self-play data
        for data in episode_data:
            buffer.add(*data)  # Add the data to the replay buffer

        if len(buffer.buffer) < 256:
            # Skip training if there isn't enough data in the buffer
            continue

        # Sample a batch of data from the replay buffer
        states, policies, values = buffer.sample(128)
        net.train()  # Set the network to training mode
        pred_policies, pred_values = net(states)  # Predict policies and values

        # Compute the loss for value and policy predictions
        value_loss = nn.MSELoss()(pred_values, values)
        policy_loss = -(policies * pred_policies).sum(dim=1).mean()
        loss = value_loss + policy_loss  # Combine the losses

        # Perform backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            # Save the model and print progress every 50 episodes
            print(f"Episode {episode} | Loss: {loss.item():.4f}")
            torch.save(net.state_dict(), f"checkpoint_{episode}.pt")

if __name__ == '__main__':
    train()  # Start the training process
