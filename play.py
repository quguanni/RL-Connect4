import torch
import numpy as np
from environment.connect4_env import Connect4
from models.alphazero_net import AlphaZeroNet
from mcts.mcts import run_mcts


def play():
    net = AlphaZeroNet()
    net.load_state_dict(torch.load("checkpoint_1000.pt", map_location=torch.device('cpu')))
    net.eval()

    env = Connect4()
    state = env.reset()
    env.render()

    while True:
        user_move = int(input("Your move (0-6): "))
        if user_move not in env.available_actions():
            print("Invalid move. Try again.")
            continue

        state, reward, done = env.step(user_move)
        env.render()
        if done:
            print("You win!" if reward == 1 else "Draw!" if reward == 0 else "Agent wins!")
            break

        pi = run_mcts(env.clone(), net, simulations=200)
        agent_move = np.argmax(pi)
        print(f"Agent chooses column {agent_move}")
        state, reward, done = env.step(agent_move)
        env.render()
        if done:
            print("Agent wins!" if reward == 1 else "Draw!" if reward == 0 else "You win!")
            break


if __name__ == '__main__':
    play()
