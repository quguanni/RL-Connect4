import numpy as np
import math
import torch

class Node:
    def __init__(self, parent, prior):
        self.parent = parent
        self.children = {}  # action -> Node
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            ucb = child.value() + c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        return best_action, best_child

def run_mcts(env, net, simulations=50):
    root = Node(parent=None, prior=1.0)
    state = env.get_state()
    policy, _ = net(torch.tensor(state).unsqueeze(0))
    policy = policy.exp().squeeze(0).detach().numpy()

    valid_moves = env.available_actions()
    policy = np.array([policy[a] if a in valid_moves else 0 for a in range(7)])
    policy_sum = np.sum(policy)
    if policy_sum > 0:
        policy /= policy_sum
    else:
        policy = np.array([1/len(valid_moves) if a in valid_moves else 0 for a in range(7)])

    for action in valid_moves:
        root.children[action] = Node(parent=root, prior=policy[action])

    for _ in range(simulations):
        node = root
        sim_env = env.clone()

        search_path = [node]

        while node.expanded():
            action, node = node.select_child()
            _, _, done = sim_env.step(action)
            search_path.append(node)
            if done:
                break

        if not done:
            state = sim_env.get_state()
            policy, value = net(torch.tensor(state).unsqueeze(0))
            policy = policy.exp().squeeze(0).detach().numpy()

            valid_moves = sim_env.available_actions()
            policy = np.array([policy[a] if a in valid_moves else 0 for a in range(7)])
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy /= policy_sum
            else:
                policy = np.array([1/len(valid_moves) if a in valid_moves else 0 for a in range(7)])

            node.children = {a: Node(parent=node, prior=policy[a]) for a in valid_moves}
        else:
            value = 0  # draw or terminal state

        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    visit_counts = np.array([root.children[a].visit_count if a in root.children else 0 for a in range(7)])
    if np.sum(visit_counts) == 0:
        action_probs = np.array([1/7] * 7)
    else:
        action_probs = visit_counts / np.sum(visit_counts)
    return action_probs