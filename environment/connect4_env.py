# connect4_env.py
import numpy as np

class Connect4:
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = 1  # Player 1 starts
    
    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = 1
        return self.board.copy()
    
    def available_moves(self):
        return [col for col in range(self.columns) if self.board[0, col] == 0]
    
    def make_move(self, col):
        for row in reversed(range(self.rows)):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                break
    
    def switch_player(self):
        self.current_player = 3 - self.current_player
    
    def check_win(self):
        # Check horizontal, vertical, diagonal for 4-in-a-row
        board = self.board
        for c in range(self.columns - 3):
            for r in range(self.rows):
                if board[r, c] == self.current_player and all(board[r, c+i] == self.current_player for i in range(4)):
                    return True

        for c in range(self.columns):
            for r in range(self.rows - 3):
                if board[r, c] == self.current_player and all(board[r+i, c] == self.current_player for i in range(4)):
                    return True

        for c in range(self.columns - 3):
            for r in range(self.rows - 3):
                if board[r, c] == self.current_player and all(board[r+i, c+i] == self.current_player for i in range(4)):
                    return True
                if board[r+3, c] == self.current_player and all(board[r+3-i, c+i] == self.current_player for i in range(4)):
                    return True
        return False
