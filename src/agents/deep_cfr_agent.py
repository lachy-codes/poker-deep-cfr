from src.memory.reservoir_memory import ReservoirMemory
from src.networks.deep_poker_network import DeepPokerNN
from torch.optim.adam import Adam
import torch
import numpy as np

class DeepCFRAgent: 
    def __init__(self, reservoir_capacity:int=300_000, num_actions:int=10, device='cpu'):
        self.memory = ReservoirMemory(reservoir_capacity)
        self.adv_net = DeepPokerNN(card_groups=4, bet_features=108, actions=10, hidden_dim=256).to(device)
        self.optimizer = Adam(self.adv_net.parameters(), lr=1e-6, weight_decay=1e-5)
        self.num_actions = num_actions
    def train_advantage_network(self, epochs, batch_size): 
        if self.memory.length < batch_size:
            return 0.0
        
        
    def regret_matched_strategy(self, card_groups, features, legal_actions): 
        # Get network predictions
        with torch.no_grad():
            advantages, bet_size = self.adv_net(card_groups, features)
            advantages = advantages[0].cpu().numpy()
            bet_size = bet_size[0][0].item()
        
        # Mask advantages
        masked_advantages = np.zeros(self.num_actions)
        for a in legal_actions:
            masked_advantages[a] = max(0, advantages[a])
        
        # Regret match
        if sum(masked_advantages) > 0:
            strategy = masked_advantages / sum(masked_advantages)
        else: 
            strategy = np.zeros(self.num_actions)
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)
        
        return strategy, bet_size
    
    