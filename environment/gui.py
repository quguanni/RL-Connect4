import pygame
import numpy as np
from .connect4_env import Connect4

class Connect4GUI:
    def __init__(self, env):
        self.env = env
        self.rows, self.cols = env.rows, env.cols
        self.cell_size = 100  # Increased cell size for better visibility
        self.width = self.cols * self.cell_size
        self.height = (self.rows + 1) * self.cell_size
        self.radius = int(self.cell_size/2 - 5)
        
        # Enhanced color palette
        self.BOARD_COLOR = (13, 71, 161)  # Deep blue
        self.BACKGROUND = (25, 118, 210)  # Lighter blue
        self.BLACK = (33, 33, 33)  # Dark gray
        self.RED = (244, 67, 54)  # Bright red
        self.YELLOW = (255, 235, 59)  # Bright yellow
        self.WHITE = (255, 255, 255)
        self.GRAY = (158, 158, 158)
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Connect 4 - AlphaZero")
        
        # Load and scale background image if available
        try:
            self.background = pygame.image.load("assets/background.jpg")
            self.background = pygame.transform.scale(self.background, (self.width, self.height))
        except:
            self.background = None
        
        # Enhanced fonts
        self.title_font = pygame.font.SysFont("Arial", 80, bold=True)
        self.font = pygame.font.SysFont("Arial", 60, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 30)
        
        # Animation properties
        self.animation_speed = 15
        self.falling_pieces = []
        
    def draw_board(self):
        # Draw background
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill(self.BACKGROUND)
        
        # Draw the board
        board_rect = pygame.Rect(0, self.cell_size, self.width, self.height - self.cell_size)
        pygame.draw.rect(self.screen, self.BOARD_COLOR, board_rect)
        
        # Draw the circles with shadow effect
        for c in range(self.cols):
            for r in range(self.rows):
                center = (int(c * self.cell_size + self.cell_size/2),
                         int(r * self.cell_size + self.cell_size/2))
                
                # Draw shadow
                pygame.draw.circle(
                    self.screen,
                    self.BLACK,
                    (center[0] + 2, center[1] + 2),
                    self.radius
                )
                
                # Draw circle
                pygame.draw.circle(
                    self.screen,
                    self.BLACK,
                    center,
                    self.radius
                )
        
        # Draw the pieces with shadow effect
        for c in range(self.cols):
            for r in range(self.rows):
                if self.env.board[r][c] != 0:
                    center = (int(c * self.cell_size + self.cell_size/2),
                             int(r * self.cell_size + self.cell_size/2))
                    color = self.RED if self.env.board[r][c] == 1 else self.YELLOW
                    
                    # Draw shadow
                    pygame.draw.circle(
                        self.screen,
                        self.BLACK,
                        (center[0] + 2, center[1] + 2),
                        self.radius
                    )
                    
                    # Draw piece
                    pygame.draw.circle(
                        self.screen,
                        color,
                        center,
                        self.radius
                    )
                    
                    # Draw highlight
                    pygame.draw.circle(
                        self.screen,
                        self.WHITE,
                        (center[0] - self.radius//3, center[1] - self.radius//3),
                        self.radius//4
                    )
        
        # Draw the hovering piece with animation
        posx = pygame.mouse.get_pos()[0]
        if self.env.current_player == 1:
            color = self.RED
        else:
            color = self.YELLOW
            
        # Draw shadow for hovering piece
        pygame.draw.circle(
            self.screen,
            self.BLACK,
            (posx + 2, int(self.cell_size/2) + 2),
            self.radius
        )
        
        # Draw hovering piece
        pygame.draw.circle(
            self.screen,
            color,
            (posx, int(self.cell_size/2)),
            self.radius
        )
        
        # Draw highlight for hovering piece
        pygame.draw.circle(
            self.screen,
            self.WHITE,
            (posx - self.radius//3, int(self.cell_size/2) - self.radius//3),
            self.radius//4
        )
        
        # Draw column numbers
        for c in range(self.cols):
            text = self.small_font.render(str(c), True, self.WHITE)
            text_rect = text.get_rect(center=(c * self.cell_size + self.cell_size/2, self.cell_size/2))
            self.screen.blit(text, text_rect)
        
        pygame.display.update()
    
    def animate_piece_drop(self, col, row):
        start_y = 0
        end_y = row * self.cell_size + self.cell_size/2
        current_y = start_y
        color = self.RED if self.env.current_player == 1 else self.YELLOW
        
        while current_y < end_y:
            current_y += self.animation_speed
            if current_y > end_y:
                current_y = end_y
                
            # Redraw the board
            self.draw_board()
            
            # Draw the falling piece
            center_x = col * self.cell_size + self.cell_size/2
            pygame.draw.circle(
                self.screen,
                color,
                (int(center_x), int(current_y)),
                self.radius
            )
            
            pygame.display.update()
            pygame.time.wait(10)
    
    def get_column_from_mouse(self, posx):
        return int(posx // self.cell_size)
    
    def show_winner(self, winner):
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))
        
        if winner == 1:
            label = self.title_font.render("Red Wins!", 1, self.RED)
        else:
            label = self.title_font.render("Yellow Wins!", 1, self.YELLOW)
        
        label_rect = label.get_rect(center=(self.width/2, self.height/2))
        self.screen.blit(label, label_rect)
        
        # Add "Click to play again" text
        restart_text = self.small_font.render("Click to play again", 1, self.WHITE)
        restart_rect = restart_text.get_rect(center=(self.width/2, self.height/2 + 80))
        self.screen.blit(restart_text, restart_rect)
        
        pygame.display.update()
        pygame.time.wait(3000)
    
    def show_draw(self):
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))
        
        label = self.title_font.render("Game Draw!", 1, self.WHITE)
        label_rect = label.get_rect(center=(self.width/2, self.height/2))
        self.screen.blit(label, label_rect)
        
        # Add "Click to play again" text
        restart_text = self.small_font.render("Click to play again", 1, self.WHITE)
        restart_rect = restart_text.get_rect(center=(self.width/2, self.height/2 + 80))
        self.screen.blit(restart_text, restart_rect)
        
        pygame.display.update()
        pygame.time.wait(3000)
    
    def close(self):
        pygame.quit() 