import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.connect4_env import Connect4
from models.alphazero_net import AlphaZeroNet
from mcts.mcts import MCTS
import torch
import numpy as np

def train():
    env = Connect4()
    net = AlphaZeroNet()
    
    # Example: Run one training step (expand later)
    state = env.reset()
    done = False
    
    while not done:
        state_tensor = torch.tensor(np.stack([(state==1), (state==2)])).float().unsqueeze(0)
        action_probs = MCTS(state_tensor.numpy(), net, simulations=10)
        
        action = np.argmax(action_probs)
        env.make_move(action)
        
        if env.check_win() or len(env.available_moves()) == 0:
            done = True
        else:
            env.switch_player()
        state = env.board.copy()
    
    print("Training step completed.")

if __name__ == "__main__":
    train()
