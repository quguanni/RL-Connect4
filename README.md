# Reinforcement Learning for Connect 4 (AlphaZero-style)

Train an agent to play Connect 4 using **Deep Reinforcement Learning**, **Monte Carlo Tree Search (MCTS)**, and **self-play** — inspired by DeepMind's AlphaZero.

---

## Project Overview

This project develops an intelligent Connect 4 agent that:
- Learns by playing **against itself**
- Uses a **neural network** to predict good moves and game outcomes
- Simulates future moves using **MCTS**
- Can also **learn from you** (human-in-the-loop)

---

## Why Connect 4?

Connect 4 is a perfect environment to study reinforcement learning because:
- Simple rules, but complex strategy
- Fully observable, turn-based, deterministic
- Fast to simulate and easy to visualize
- Solved game — perfect for benchmarking AI skill

---

## Methodology

### Components:
- **Environment:** A full simulation of Connect 4 (`connect4_env.py`)
- **Neural Network:** A dual-headed CNN (`alphazero_net.py`) that:
  - Predicts **move probabilities** (policy head)
  - Predicts **expected outcome** (value head)
- **Planner:** Monte Carlo Tree Search (`mcts.py`) uses the network to plan moves
- **Trainer:** Self-play loop (`train.py`) trains the agent using its own experience

### Learning:
- Every game is stored in a replay buffer
- The model is trained to match MCTS policies and predict win/loss outcomes
- Over time, the agent becomes harder to beat — and can even learn from humans

---

### Repository Structure

```plaintext
RL-Connect4/
├── environment/
│   └── connect4_env.py        # Game engine and rules
├── models/
│   └── alphazero_net.py       # Neural network (policy + value heads)
├── mcts/
│   └── mcts.py                # Monte Carlo Tree Search logic
├── training/
│   └── train.py               # Self-play + human-in-the-loop training loop
├── play.py                    # Play against the trained agent via terminal
├── checkpoint_*.pt            # Saved neural network weights
├── training_log.txt           # (optional) Saved training progress
└── README.md
```

### How to Play

After training, run:
(`python play.py`)

### How to Train

- To train via self-play:

(`python training/train.py`)

- To train via human games:

(`python training/train.py`)
Then type 2 when prompted

### Checkpoints
- Model is saved every 50 episodes as checkpoint_50.pt, checkpoint_100.pt, etc.
- (`play.py`) automatically loads the latest one.
- - Policy loss: trains the network to match MCTS probabilities.
- - Value loss: trains it to predict who wins from any board.