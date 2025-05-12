import numpy as np

class Connect4:
    def __init__(self):
        self.rows, self.cols = 6, 7
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.get_state()

    def clone(self):
        clone = Connect4()
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        return clone

    def available_actions(self):
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def step(self, action):
        if self.board[0, action] != 0:
            raise ValueError("Invalid action: column is full")

        for row in reversed(range(self.rows)):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break

        done = self.check_winner(self.current_player)
        draw = np.all(self.board != 0)

        reward = 1 if done else 0
        self.current_player *= -1

        return self.get_state(), reward, done or draw

    def get_state(self):
        p1 = (self.board == 1).astype(np.float32)
        p2 = (self.board == -1).astype(np.float32)
        return np.array([p1, p2])

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
        print('0 1 2 3 4 5 6\n')
