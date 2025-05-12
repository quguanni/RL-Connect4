import torch
import numpy as np
import os
import glob
from environment.connect4_env import Connect4
from models.alphazero_net import AlphaZeroNet
from mcts.mcts import run_mcts


def play():
    # Automatically load latest checkpoint
    checkpoints = sorted(glob.glob("checkpoint_*.pt"))
    if not checkpoints:
        print("No checkpoint found. Please train the model first.")
        return

    latest = checkpoints[-1]
    print(f"Loading model from {latest}...")
    net = AlphaZeroNet()
    net.load_state_dict(torch.load(latest, map_location='cpu'))
    net.eval()

    env = Connect4()
    state = env.reset()
    env.render()

    while True:
        try:
            user_move = int(input("Your move (0-6): "))
        except ValueError:
            print("Please enter a valid number between 0 and 6.")
            continue

        if user_move not in env.available_actions():
            print("Invalid move. Try again.")
            continue

        state, reward, done = env.step(user_move)
        env.render()
        if done:
            print("You win!" if reward == 1 else "Draw!")
            break

        # Agent plays
        pi = run_mcts(env.clone(), net, simulations=200)
        if sum(env.board.flatten() == 0) < 14:
            agent_move = np.argmax(pi)
        else:
            agent_move = np.random.choice(len(pi), p=pi)

        print(f"Agent chooses column {agent_move}")
        state, reward, done = env.step(agent_move)
        env.render()
        if done:
            print("Agent wins!" if reward == 1 else "Draw!" if reward == 0 else "You win!")
            break


if __name__ == '__main__':
    play()
