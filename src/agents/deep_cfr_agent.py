from src.memory.reservoir_memory import ReservoirMemory
from src.networks.deep_poker_network import DeepPokerNN
from torch.optim.adam import Adam
import torch
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import List, Tuple, Dict, Optional

class DeepCFRAgent: 
    def __init__(self, reservoir_capacity:int=3_000_000, num_actions:int=3, device='cpu'):
        self.memory = ReservoirMemory(reservoir_capacity)
        self.adv_net = DeepPokerNN(card_groups=4, bet_features=108, actions=3).to(device)
        
        self.adv_net.eval()
        
        self.optimizer = Adam(self.adv_net.parameters(), lr=1e-5, weight_decay=1e-5)
        self.num_actions = num_actions
        
        
    def train_advantage_network(
        self,
        batch_size: int = 10_000,
        epochs: int = 1,
        sizing_loss_weight: float = 0.05,           # tune between 0.05–0.3
        grad_clip_max_norm: float = 1.0,
        use_amp: bool = True,
        scaler: Optional[GradScaler] = None
    ) -> Dict[str, float]:
        """
        Trains the advantage network using uniform reservoir sampling.
        
        Assumes:
        - self.advantage_memory.sample(batch_size) returns list of tuples:
        (card_groups, bet_features, action_idx, bet_size, weighted_regret)
        - action_idx: 0=fold, 1=check/call, 2=raise
        - bet_size: already normalized to [0, 1]
        """
        # self.adv_net = DeepPokerNN(card_groups=4, bet_features=108, actions=3).to('mps')
        # self.optimizer = Adam(self.adv_net.parameters(), lr=1e-5, weight_decay=1e-5)
        self.adv_net = self.adv_net.to('mps')
        
        if self.memory.length < batch_size:
            return {"total_loss": 0.0, "adv_loss": 0.0, "size_loss": 0.0}

        self.device = torch.device('mps') if torch.mps.is_available() else torch.device('cpu')
            
        self.adv_net.train()

        total_adv_loss = 0.0
        total_size_loss = 0.0
        total_steps = 0

        for epoch in range(epochs):
            # Uniform sampling — no priorities, no weights
            batch: List[Tuple] = self.memory.sample_batch(batch_size)

            # Efficient collation
            card_groups_list = [[] for _ in range(4)]  # hole, flop, turn, river
            bet_features_list = []
            action_idx_list = []
            bet_size_list = []
            target_regret_list = []

            for cg, bf, a_idx, b_size, regret in batch:
                for i, g in enumerate(cg):
                    card_groups_list[i].append(g)
                bet_features_list.append(bf)
                action_idx_list.append(a_idx)
                bet_size_list.append(b_size)
                target_regret_list.append(regret)

            # Stack tensors
            card_groups_batch = [torch.cat(lst, dim=0).to('mps') for lst in card_groups_list]
            bet_features_batch = torch.cat(bet_features_list, dim=0).to(self.device)

            action_idx_batch   = torch.tensor(action_idx_list,   dtype=torch.long,   device=self.device)
            bet_size_batch     = torch.tensor(bet_size_list,     dtype=torch.float32, device=self.device)
            target_regret_batch = torch.tensor(target_regret_list, dtype=torch.float32, device=self.device)

            # Forward + loss
            self.optimizer.zero_grad(set_to_none=True)

            if use_amp and scaler is not None:
                with autocast(device_type=self.device.type):
                    action_logits, pred_bet_frac = self.adv_net(card_groups_batch, bet_features_batch)

                    # Advantage loss
                    chosen_adv = action_logits[range(batch_size), action_idx_batch]
                    adv_loss = F.smooth_l1_loss(chosen_adv, target_regret_batch, reduction='mean')

                    # Sizing loss — only on raise actions
                    is_raise = (action_idx_batch == 2)
                    size_loss = torch.tensor(0.0, device=self.device)
                    if is_raise.any():
                        pred_frac = pred_bet_frac.squeeze(-1)[is_raise].detach()
                        true_frac = bet_size_batch[is_raise]
                        size_loss = F.smooth_l1_loss(pred_frac, true_frac, reduction='mean')

                    total_loss = adv_loss + sizing_loss_weight * size_loss

                scaler.scale(total_loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.adv_net.parameters(), max_norm=0.1)
                scaler.step(self.optimizer)
                scaler.update()

            else:
                # No AMP
                action_logits, pred_bet_frac = self.adv_net(card_groups_batch, bet_features_batch)

                chosen_adv = action_logits[range(batch_size), action_idx_batch]
                adv_loss = F.smooth_l1_loss(chosen_adv, target_regret_batch, reduction='mean')

                is_raise = (action_idx_batch == 2)
                size_loss = torch.tensor(0.0, device=self.device)
                if is_raise.any():
                    pred_frac = pred_bet_frac.squeeze(-1)[is_raise].detach()
                    true_frac = bet_size_batch[is_raise]
                    regret = target_regret_batch[is_raise]
                    
                    weight = torch.clamp(regret, min=0.0) + 0.1
                    
                    size_loss = (weight * F.mse_loss(pred_frac, true_frac, reduction='none')).mean()

                total_loss = adv_loss + sizing_loss_weight * size_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adv_net.parameters(), max_norm=grad_clip_max_norm)
                self.optimizer.step()

            total_adv_loss += adv_loss.item()
            total_size_loss += size_loss.item()
            total_steps += 1

        avg_adv_loss = total_adv_loss / total_steps
        avg_size_loss = total_size_loss / total_steps
        
        self.adv_net.to('cpu')
        
        return {
            "total_loss":   avg_adv_loss + sizing_loss_weight * avg_size_loss,
            "adv_loss":     avg_adv_loss,
            "size_loss":    avg_size_loss,
            "steps":        total_steps,
            "batch_size":   batch_size,
            "epochs":       epochs,
        }

    def regret_matched_strategy(self, card_groups, features, legal_actions): 
        if next(self.adv_net.parameters()).device != 'cpu':
            self.adv_net = self.adv_net.to('cpu')
        
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
    
