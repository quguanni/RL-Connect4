import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os

# Connect4 Environment
class Connect4:
    def __init__(self):
        self.rows, self.cols = 6, 7
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.board.copy()

    def available_actions(self):
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def step(self, action):
        for row in reversed(range(self.rows)):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break

        done = self.check_winner(self.current_player)
        draw = np.all(self.board != 0)

        reward = 1 if done else (0.5 if draw else 0)
        self.current_player *= -1

        return self.board.copy(), reward, done or draw

    def check_winner(self, player):
        for c in range(self.cols - 3):
            for r in range(self.rows):
                if all(self.board[r, c+i] == player for i in range(4)):
                    return True
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if all(self.board[r+i, c] == player for i in range(4)):
                    return True
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                if all(self.board[r+i, c+i] == player for i in range(4)):
                    return True
        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                if all(self.board[r-i, c+i] == player for i in range(4)):
                    return True
        return False

    def render(self):
        symbols = {0: '⚪', 1: '🔴', -1: '🟡'}
        for row in self.board:
            print(' '.join(symbols[x] for x in row))
        print('0  1  2  3  4  5  6\n')

# Deep Q-Network
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)

    def forward(self, x):
        x = x.view(-1, 6 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Training with Self-Play
def train_self_play():
    env = Connect4()
    model = DQN()
    target_model = DQN()
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    replay_buffer = deque(maxlen=50000)

    episodes = 10000
    batch_size = 128
    gamma = 0.99
    epsilon, epsilon_decay, epsilon_min = 1.0, 0.999, 0.05
    target_update_freq = 50

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            current_player = env.current_player
            valid_actions = env.available_actions()
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                state_tensor = torch.tensor(state * current_player, dtype=torch.float32).unsqueeze(0)
                q_values = model(state_tensor).detach().numpy()[0]
                q_values = [q_values[a] if a in valid_actions else -np.inf for a in range(7)]
                action = np.argmax(q_values)

            next_state, reward, done = env.step(action)
            reward = reward if done else 0

            replay_buffer.append((state * current_player, action, reward, next_state * current_player, done))

            state = next_state

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = target_model(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

                loss = F.mse_loss(q_values, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % 100 == 0:
            print(f"Episode {episode}, epsilon={epsilon:.3f}")

    torch.save(model.state_dict(), "connect4_selfplay.pt")
    print("Training complete, model saved.")

# Play against trained agent
def play():
    env = Connect4()
    model = DQN()
    model.load_state_dict(torch.load("connect4_selfplay.pt", weights_only=True))
    model.eval()

    state = env.reset()
    env.render()

    while True:
        move = int(input("Your move (0-6): "))
        if move not in env.available_actions():
            print("Invalid move. Try again.")
            continue

        state, reward, done = env.step(move)
        env.render()
        if done:
            print("You win!" if reward == 1 else "Draw!")
            break

        state_tensor = torch.tensor(-state, dtype=torch.float32).unsqueeze(0)  # Fixed line here (-state)
        q_values = model(state_tensor).detach().numpy()[0]
        q_values = [q_values[a] if a in env.available_actions() else -np.inf for a in range(7)]
        agent_move = np.argmax(q_values)

        state, reward, done = env.step(agent_move)
        print("Agent's move:")
        env.render()
        if done:
            print("Agent wins!" if reward == 1 else "Draw!")
            break

if __name__ == '__main__':
    if os.path.exists("connect4_selfplay.pt"):
        play()
    else:
        train_self_play()
