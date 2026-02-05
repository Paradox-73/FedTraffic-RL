import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class TrafficLightDQN(nn.Module):
    def __init__(self, state_dim=11, action_dim=2):
        super(TrafficLightDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TrafficLightAgent:
    def __init__(self, state_dim=11, action_dim=2, device="cpu", lr=1e-4, gamma=0.99, batch_size=64, memory_size=10000):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.model = TrafficLightDQN(state_dim, action_dim).to(self.device)
        self.target_model = TrafficLightDQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5) # Lower LR for stability
        self.memory = deque(maxlen=memory_size)
        self.loss_fn = nn.HuberLoss() # Better at handling the "spiky" rewards of Stage 3

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return torch.argmax(q_values, dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())