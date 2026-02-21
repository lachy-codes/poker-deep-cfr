import torch.nn as nn
import torch
import torch.nn.functional as F

class CardEmbedding(nn.Module):
    """
    Embedding for a group of cards (e.g. hole cards, flop, turn, river).
    
    For each card we learn three embeddings:
      - rank embedding (13 possibilities)
      - suit embedding (4 possibilities)
      - full card embedding (52 possibilities)
    
    We sum them per card, then sum across all cards in the group.
    This makes the representation order-invariant within the group.
    
    Input shape:  (batch_size, num_cards_in_group)  with values in [0, 51]
                  (0 = 2♣, 1 = 3♣, ..., 12 = A♣, 13 = 2♦, ..., 51 = A♠)
    Output shape: (batch_size, embedding_dim)
    """
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Three separate embedding tables
        self.rank_emb   = nn.Embedding(num_embeddings=13,  embedding_dim=embedding_dim)
        self.suit_emb   = nn.Embedding(num_embeddings=4,   embedding_dim=embedding_dim)
        self.card_emb   = nn.Embedding(num_embeddings=52,  embedding_dim=embedding_dim)
        
        # Optional: small MLP after summing (helps expressiveness without much cost)
        self.post_sum = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        """
        cards: torch.LongTensor of shape (batch_size, num_cards)
               values ∈ [0, 51]  (standard 52-card deck indexing)
               -1 can be used for missing cards (masked out)
        """
        
        # Mask for missing cards (e.g. preflop has no turn/river)
        valid = (cards >= 0)  # shape (B, num_cards)
        
        # Compute rank and suit from card id
        ranks = cards % 13   # 0..12
        suits = cards // 13  # 0..3
        
        # Get embeddings
        rank_e = self.rank_emb(ranks)    # (B, num_cards, dim)
        suit_e = self.suit_emb(suits)    # (B, num_cards, dim)
        card_e = self.card_emb(cards.clamp(min=0))  # (B, num_cards, dim)  # clamp to avoid -1 index error
        
        # Sum the three embeddings per card
        per_card = rank_e + suit_e + card_e   # (B, num_cards, dim)
        
        # Mask out invalid (padding) cards
        per_card = per_card * valid.unsqueeze(-1).float()
        
        # Sum over cards → permutation invariant
        summed = per_card.sum(dim=1)          # (B, dim)
        
        # projection (helps when groups have very different sizes)
        out = self.post_sum(summed)
        
        return out

class DeepPokerNN(nn.module):
    def __init__(self, card_groups, bet_features, actions, hidden_dim):
        super().__init__()
        
        # Card branch
        self.card_embs = nn.ModuleList([CardEmbedding(hidden_dim) for _ in range(card_groups)])
        self.card1 = nn.Linear(hidden_dim * card_groups, hidden_dim)
        self.card2 = nn.Linear(hidden_dim, hidden_dim)
        self.card3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Bet branch
        self.bet1 = nn.Linear(bet_features * 2, hidden_dim)   # size + occurred
        self.bet2 = nn.Linear(hidden_dim, hidden_dim)                  # residual
        
        # Combined trunk (3 layers with residuals)
        self.comb1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.comb2 = nn.Linear(hidden_dim, hidden_dim)
        self.comb3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.action_head = nn.Linear(hidden_dim, actions)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, card_groups, bets):
        # cards_groups = list of tensors: [hole (B,2), flop (B,3), turn (B,1), river (B,1)]
        card_feats = [emb(g) for emb, g in zip(self.card_embs, card_groups)]
        card_feats = torch.cat(card_feats, dim=1)
        x = F.relu(self.card1(card_feats))
        x = F.relu(self.card2(x))
        x = F.relu(self.card3(x))
        
        # Bet branch
        bet_size = bets.clamp(0, 1e6)
        bet_occur = (bets >= 0).float()
        bet_in = torch.cat([bet_size, bet_occur], dim=1)
        y = F.relu(self.bet1(bet_in))
        y = F.relu(self.bet2(y) + y)   # residual
        
        # Merge + trunk
        z = torch.cat([x, y], dim=1)
        z = F.relu(self.comb1(z))
        z = F.relu(self.comb2(z) + z)
        z = F.relu(self.comb3(z) + z)
        z = self.norm(z)
        
        return self.action_head(z)   # raw logits → advantages or strategy
    
    