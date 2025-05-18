import pygame
import sys
from environment.connect4_env import Connect4
from environment.gui import Connect4GUI
from models.alphazero_net import AlphaZeroNet
from mcts.mcts import run_mcts
import torch

def play_game():
    # Initialize the environment and GUI
    env = Connect4()
    gui = Connect4GUI(env)
    
    # Load the trained model
    net = AlphaZeroNet()
    try:
        net.load_state_dict(torch.load("checkpoint_1000.pt"))
        print("Loaded model from checkpoint_1000.pt")
    except:
        print("No model checkpoint found. Starting with untrained model.")
    
    def reset_game():
        nonlocal env
        env = Connect4()
        gui.env = env
        gui.draw_board()
    
    # Game loop
    while True:
        # Human player's turn
        if env.current_player == 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    gui.close()
                    sys.exit()
                
                if event.type == pygame.MOUSEMOTION:
                    gui.draw_board()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Get the column from mouse position
                    posx = event.pos[0]
                    col = gui.get_column_from_mouse(posx)
                    
                    if col in env.available_actions():
                        # Find the row where the piece will land
                        for r in reversed(range(env.rows)):
                            if env.board[r][col] == 0:
                                # Animate the piece drop
                                gui.animate_piece_drop(col, r)
                                break
                        
                        state, reward, done = env.step(col)
                        gui.draw_board()
                        
                        if done:
                            if reward == 1:
                                gui.show_winner(1)
                            else:
                                gui.show_draw()
                            # Wait for click to restart
                            waiting_for_restart = True
                            while waiting_for_restart:
                                for event in pygame.event.get():
                                    if event.type == pygame.QUIT:
                                        gui.close()
                                        sys.exit()
                                    if event.type == pygame.MOUSEBUTTONDOWN:
                                        waiting_for_restart = False
                                        reset_game()
                                        break
        
        # AI player's turn
        else:
            # Run MCTS to get action probabilities
            action_probs = run_mcts(env, net)
            # Choose the action with highest probability
            action = action_probs.argmax()
            
            # Find the row where the piece will land
            for r in reversed(range(env.rows)):
                if env.board[r][action] == 0:
                    # Animate the piece drop
                    gui.animate_piece_drop(action, r)
                    break
            
            state, reward, done = env.step(action)
            gui.draw_board()
            
            if done:
                if reward == 1:
                    gui.show_winner(-1)
                else:
                    gui.show_draw()
                # Wait for click to restart
                waiting_for_restart = True
                while waiting_for_restart:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            gui.close()
                            sys.exit()
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            waiting_for_restart = False
                            reset_game()
                            break

if __name__ == "__main__":
    play_game() 