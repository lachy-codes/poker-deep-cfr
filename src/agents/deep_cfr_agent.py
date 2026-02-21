from memory.reservoir_memory import ReservoirMemory
from networks.deep_poker_network import DeepPokerNN
from torch.optim.adam import Adam

class DeepCFRAgent: 
    def __init__(self, reservoir_capacity:int=300_000, device='cpu'):
        self.memory = ReservoirMemory(reservoir_capacity)
        self.adv_net = DeepPokerNN(card_groups=4, bet_features=108, actions=10, hidden_dim=256).to(device)
        self.optimizer = Adam(self.adv_net.parameters(), lr=1e-6, weight_decay=1e-5)
        
    def train_advantage_network(self, epochs, batch_size): 
        if self.memory.length < batch_size:
            return 0.0
        
        
        