import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
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

def self_play_episode(env, net, simulations=100, c_puct=1.5, strong=False):
    history = []
    env.reset()
    done = False
    moves = 0

    while not done:
        state = env.get_state()
        pi = run_mcts(env.clone(), net, simulations=simulations, c_puct=c_puct)
        action = np.random.choice(7, p=pi)
        history.append((state, pi))
        _, reward, done = env.step(action)
        moves += 1

    if strong and reward == 1:
        z = 1.18 - (9 * moves / 350)
    else:
        z = reward

    return [(s, p, z if i % 2 == 0 else -z) for i, (s, p) in enumerate(history)]

def train(start_episode=1, total_episodes=None):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    if total_episodes is None:
        total_episodes = config.get("episodes", 1000)
    net = AlphaZeroNet()

    # Load latest checkpoint if available
    checkpoints = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pt")])
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Loading {latest}...")
        net.load_state_dict(torch.load(os.path.join("checkpoints", latest)))
        start_episode = int(latest.split("_")[1].split(".")[0]) + 1

    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=config["replay_buffer_size"])

    from tqdm import tqdm

    for episode in tqdm(range(start_episode, start_episode + total_episodes), desc="Training", unit="ep"):
        env = Connect4()
        episode_data = self_play_episode(env, net, simulations=100, c_puct=1.5, strong=True)
        for data in episode_data:
            buffer.add(*data)

        if len(buffer.buffer) < 256:
            continue

        batch_size = config.get("batch_size", 128)
        states, policies, values = buffer.sample(batch_size)
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
            torch.save(net.state_dict(), os.path.join("checkpoints", f"checkpoint_{episode}.pt"))

        # Update config.yaml with new start_episode and batch_size
        config["start_episode"] = episode + 1
        with open("config.yaml", "w") as f:
            yaml.dump(config, f)

def play_against_agent(net):
    env = Connect4()
    env.reset()
    env.render()
    done = False

    while not done:
        # Human move
        try:
            col = int(input("Your move (0–6): "))
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 6.")
            continue
        if col not in env.available_actions():
            print("Column full or invalid.")
            continue
        _, reward, done = env.step(col)
        env.render()
        if done:
            print("You win!" if reward == 1 else "Draw!")
            break

        # Agent move
        pi = run_mcts(env.clone(), net, simulations=100, c_puct=1.5)
        ai_move = np.argmax(pi)
        print(f"Agent chooses column {ai_move}")
        _, reward, done = env.step(ai_move)
        env.render()
        if done:
            print("Agent wins!" if reward == 1 else "Draw!")
            break

if __name__ == '__main__':
    os.makedirs("checkpoints", exist_ok=True)
    mode = input("Choose mode: (1) train, (2) play: ").strip()
    if mode == '1':
        train(total_episodes=1000)  # Change this number to continue training more episodes
    else:
        net = AlphaZeroNet()
        latest_ckpt = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pt")])[-1]
        print(f"Loading {latest_ckpt}...")
        net.load_state_dict(torch.load(os.path.join("checkpoints", latest_ckpt)))
        play_against_agent(net)
