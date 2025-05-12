import numpy as np
import torch

class Node:
    def __init__(self, state, parent=None, prior=0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.prior = prior

    def expanded(self):
        return len(self.children) > 0

    def best_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_child = None
        for action, child in self.children.items():
            ucb = child.value / (1 + child.visits) + c_puct * child.prior * np.sqrt(self.visits) / (1 + child.visits)
            if ucb > best_score:
                best_score = ucb
                best_child = (action, child)
        return best_child

def MCTS(state, net, simulations=50):
    root = Node(state)

    # CORRECTED LINE HERE:
    policy, _ = net(torch.tensor(state.squeeze(0)).float())
    policy = policy.detach().numpy().flatten()

    for action, p in enumerate(policy):
        root.children[action] = Node(state=None, parent=root, prior=p)


    for _ in range(simulations):
        node = root
        path = [node]

        # Selection
        while node.expanded():
            action, node = node.best_child()
            path.append(node)

        # Expansion
        if node.visits == 0:
            state = node.parent.state.copy()
            game = Connect4()
            game.board = state
            game.current_player = 1
            game.make_move(action)

            if game.check_win():
                value = 1  # Win condition
            elif len(game.available_moves()) == 0:
                value = 0  # Draw condition
            else:
                next_state = game.board.copy()
                node.state = next_state
                _, value = net(torch.tensor(next_state).unsqueeze(0).float())
                value = value.item()
        else:
            value = 0

        # Backpropagation
        for n in reversed(path):
            n.visits += 1
            n.value += value

    action_probs = np.zeros(7)
    for action, child in root.children.items():
        action_probs[action] = child.visits

    action_probs /= np.sum(action_probs)
    return action_probs