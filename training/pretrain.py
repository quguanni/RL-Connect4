import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.alphazero_net import AlphaZeroNet
from environment.connect4_env import Connect4

# Dummy function: Replace this with your own solver
def solve_position(board, current_player):
    # Return a one-hot move vector and value: best move is always column 3
    move = 3
    pi = np.zeros(7)
    pi[move] = 1
    value = 1  # Assume it's a win for supervised example
    return pi, value

def generate_dataset(num_samples=1000):
    boards, policies, values = [], [], []
    for _ in range(num_samples):
        env = Connect4()
        pi, z = solve_position(env.board, env.current_player)
        boards.append(env.get_state())
        policies.append(pi)
        values.append(z)
    return (
        torch.tensor(np.array(boards), dtype=torch.float32),
        torch.tensor(np.array(policies), dtype=torch.float32),
        torch.tensor(np.array(values), dtype=torch.float32)
    )

def pretrain():
    net = AlphaZeroNet()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    states, policies, values = generate_dataset()
    for epoch in range(10):
        net.train()
        pred_policies, pred_values = net(states)
        policy_loss = -(policies * pred_policies).sum(dim=1).mean()
        value_loss = nn.MSELoss()(pred_values, values)
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/10 | Loss: {loss.item():.4f}")

    torch.save(net.state_dict(), "pretrained.pt")
    print("✅ Saved pretrained model as pretrained.pt")

if __name__ == "__main__":
    pretrain()
