import torch
import numpy as np
from models.alphazero_net import AlphaZeroNet
from environment.connect4_env import Connect4

# Dummy test set (replace with real positions)
def get_test_set():
    examples = []
    for _ in range(100):
        env = Connect4()
        state = env.get_state()
        best_move = 3  # pretend best move is always 3
        examples.append((state, best_move))
    return examples

def evaluate_model(checkpoint_path):
    net = AlphaZeroNet()
    net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    net.eval()

    test_data = get_test_set()
    correct = 0
    for state, best_move in test_data:
        with torch.no_grad():
            logits, _ = net(torch.tensor(state).unsqueeze(0))
            probs = logits.exp().squeeze(0).numpy()
            if np.argmax(probs) == best_move:
                correct += 1

    acc = correct / len(test_data)
    print(f"Policy accuracy: {acc*100:.2f}% on {len(test_data)} test positions")

if __name__ == '__main__':
    evaluate_model("checkpoint_1000.pt")  # Change to any checkpoint
