# F-DQN Routing Protocol for MANETs

This repository contains the source code for the paper: **"Adaptive Load-Balancing and Energy-Aware Routing in MANETs using Federated Deep Q-Networks"**.

## Project Structure
- `env/`: MANET simulation environment and energy models.
- `agents/`: Local DQN agents and Federated Server (FedAvg) algorithms.
- `training/`: Training loop for the F-DQN protocol.
- `evaluation/`: Comparison scripts against the baseline (Shortest Path/Dijkstra).
- `results/`: Generated figures and CSV tables.

## How to Run
To run the full simulation, train the agents, and evaluate the protocol, execute the main script:
```bash
python main.py
