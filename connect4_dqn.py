import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
import pickle

# --- Connect 4 Environment ---
class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.board.copy()

    def available_actions(self):
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def step(self, action):
        for row in reversed(range(self.rows)):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                break

        done = self.check_winner(self.current_player)
        draw = np.all(self.board != 0)

        reward = 0
        if done:
            reward = 1
        elif draw:
            reward = 0.5
        else:
            reward = self.reward_shaping(self.current_player)

        self.current_player *= -1
        return self.board.copy(), reward, done or draw

    def reward_shaping(self, player):
        # +0.1 for 3 in a row
        count = 0
        for r in range(self.rows):
            for c in range(self.cols - 2):
                window = self.board[r, c:c+3]
                if np.count_nonzero(window == player) == 3 and np.count_nonzero(window == 0) == 0:
                    count += 1
        return 0.1 * count

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
        print('\n'.join([' '.join(['⚪' if x == 0 else ('🔴' if x == 1 else '🟡') for x in row]) for row in self.board]))
        print('0 1 2 3 4 5 6\n')

# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)

    def forward(self, x):
        x = x.view(-1, 6 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- Smart Opponent (blocks win if possible) ---
def smart_opponent_move(env):
    for action in env.available_actions():
        temp_env = Connect4()
        temp_env.board = env.board.copy()
        temp_env.current_player = -1
        temp_env.step(action)
        if temp_env.check_winner(-1):
            return action
    return random.choice(env.available_actions())

# --- Training Setup ---
def train():
    env = Connect4()
    model = DQN()
    target_model = DQN()
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    replay_buffer = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    update_target_every = 20

    for episode in range(5000):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            valid_actions = env.available_actions()
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = model(state_tensor)[0].detach().numpy()
                    q_values = [q_values[a] if a in valid_actions else -np.inf for a in range(7)]
                    action = int(np.argmax(q_values))

            next_state, reward, done = env.step(action)

            # Smart opponent move
            if not done:
                opp_action = smart_opponent_move(env)
                next_state, opp_reward, done = env.step(opp_action)
                reward -= opp_reward  # penalize agent if opponent gets closer to win

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = target_model(next_states).max(1)[0]
                target_q = rewards + gamma * next_q_values * (1 - dones)

                loss = F.mse_loss(q_values, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    torch.save(model.state_dict(), "connect4_model.pt")
    print("Model saved as connect4_model.pt")

# --- Play Against Trained Agent ---
def play():
    env = Connect4()
    model = DQN()
    model.load_state_dict(torch.load("connect4_model.pt"))
    model.eval()

    state = env.reset()
    env.render()

    while True:
        col = int(input("Your move (0-6): "))
        if col not in env.available_actions():
            print("Invalid move. Try again.")
            continue
        state, reward, done = env.step(col)
        env.render()
        if done:
            print("Game over! You win!" if reward == 1 else "Draw!" if reward == 0.5 else "Game over!")
            break

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state_tensor)[0].detach().numpy()
            q_values = [q_values[a] if a in env.available_actions() else -np.inf for a in range(7)]
            action = int(np.argmax(q_values))

        state, reward, done = env.step(action)
        print("Agent plays:")
        env.render()
        if done:
            print("Game over! Agent wins!" if reward == 1 else "Draw!" if reward == 0.5 else "Game over!")
            break

if __name__ == "__main__":
    if os.path.exists("connect4_model.pt"):
        play()
    else:
        train()

