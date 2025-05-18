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
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return torch.tensor(np.array(states), dtype=torch.float32), \
               torch.tensor(np.array(policies), dtype=torch.float32), \
               torch.tensor(np.array(values), dtype=torch.float32)

def self_play_episode(env, net, simulations=50):
    history = []
    env.reset()
    done = False

    while not done:
        state = env.get_state()
        pi = run_mcts(env.clone(), net, simulations)
        action = np.random.choice(7, p=pi)
        history.append((state, pi))
        _, reward, done = env.step(action)

    return [(s, p, reward if i % 2 == 0 else -reward) for i, (s, p) in enumerate(history)]

def human_play_episode(env, net, simulations=200):
    history = []
    env.reset()
    done = False

    while not done:
        env.render()
        try:
            user_move = int(input("Your move (0–6): "))
        except ValueError:
            print("Please enter a number.")
            continue
        if user_move not in env.available_actions():
            print("Invalid move.")
            continue
        state = env.get_state()
        history.append((state, np.eye(7)[user_move]))
        _, reward, done = env.step(user_move)
        if done:
            break

        state = env.get_state()
        pi = run_mcts(env.clone(), net, simulations)
        agent_move = np.argmax(pi)
        history.append((state, pi))
        _, reward, done = env.step(agent_move)

    return [(s, p, reward if i % 2 == 0 else -reward) for i, (s, p) in enumerate(history)]

def train():
    net = AlphaZeroNet()
    if os.path.exists("checkpoint_1000.pt"):
        print("Loading checkpoint_1000.pt...")
        net.load_state_dict(torch.load("checkpoint_1000.pt", weights_only=True))

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=10000)

    for episode in range(1001, 20001):
        env = Connect4()
        episode_data = self_play_episode(env, net)
        for data in episode_data:
            buffer.add(*data)

        if len(buffer.buffer) < 256:
            continue

        states, policies, values = buffer.sample(128)
        net.train()
        pred_policies, pred_values = net(states)

        value_loss = nn.MSELoss()(pred_values, values)
        policy_loss = -(policies * pred_policies).sum(dim=1).mean()
        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"Episode {episode} | Loss: {loss.item():.4f}")
            torch.save(net.state_dict(), f"checkpoint_{episode}.pt")

def train_with_human():
    net = AlphaZeroNet()
    if os.path.exists("checkpoint_1000.pt"):
        print("Loading checkpoint_1000.pt...")
        net.load_state_dict(torch.load("checkpoint_1000.pt", weights_only=True))

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=10000)

    while True:
        env = Connect4()
        print("Training on your next game...")
        episode_data = human_play_episode(env, net)
        for data in episode_data:
            buffer.add(*data)

        if len(buffer.buffer) < 256:
            continue

        for _ in range(5):  # Train multiple times on one human game
            states, policies, values = buffer.sample(128)
            net.train()
            pred_policies, pred_values = net(states)

            value_loss = nn.MSELoss()(pred_values, values)
            policy_loss = -(policies * pred_policies).sum(dim=1).mean()
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), "checkpoint_human.pt")
        print("✅ Checkpoint updated. Ready for another game!\n")

if __name__ == '__main__':
    mode = input("Train mode — (1) Self-Play or (2) Human-in-the-loop? ").strip()
    if mode == "2":
        train_with_human()
    else:
        train()

