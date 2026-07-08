import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from models.dqn import RoutingDQN
import os

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, reward, next_states, done):
        self.buffer.append((state, reward, next_states, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim=4, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = RoutingDQN(state_dim).to(self.device)
        self.target_net = RoutingDQN(state_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.criterion = nn.MSELoss()

    def select_next_hop(self, current_features, neighbor_features_list, neighbors):
        if not neighbors: return None, None

        if random.random() < self.epsilon:
            idx = random.choice(range(len(neighbors)))
            return neighbors[idx], neighbor_features_list[idx]

        best_q = -float('inf')
        best_neighbor, best_feat = None, None

        self.q_net.eval()
        with torch.no_grad():
            for i, feat in enumerate(neighbor_features_list):
                state_tensor = torch.FloatTensor(feat).unsqueeze(0).to(self.device)
                q_val = self.q_net(state_tensor).item()
                if q_val > best_q:
                    best_q = q_val
                    best_neighbor = neighbors[i]
                    best_feat = feat
        self.q_net.train()
        return best_neighbor, best_feat

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size: return 0.0

        batch = self.memory.sample(batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[1] for t in batch])).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.array([t[3] for t in batch])).unsqueeze(1).to(self.device)

        current_q = self.q_net(states)
        max_next_q = torch.zeros(batch_size, 1).to(self.device)
        with torch.no_grad():
            for i, (_, _, next_states, done) in enumerate(batch):
                if not done and next_states is not None and len(next_states) > 0:
                    next_states_tensor = torch.FloatTensor(next_states).to(self.device)
                    max_next_q[i] = self.target_net(next_states_tensor).max().unsqueeze(0)

        expected_q = rewards + (self.gamma * max_next_q * (1 - dones))
        loss = self.criterion(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_weights(self): return self.q_net.state_dict()
    def set_weights(self, weights):
        self.q_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)

    def save_model(self, path="project/saved_models/global_model.pth"):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path="project/saved_models/global_model.pth"):
        if os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.q_net.state_dict())
