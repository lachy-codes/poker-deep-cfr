import pokers as pkrs
import torch

def encode_state(
    state: pkrs.State,
    action_history: list[tuple[str, float]],
    pot_before_action: list[float],
    max_history_actions: int=24,
    device: torch.device = torch.device('cpu')
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Encodes ONE infoset using the official pokers.State object.
    
    Traversal loop maintains two small lists:
        action_history and pot_before_action
    
    Returns:
        cards_groups : List[torch.Tensor]  [ (1,2), (1,3), (1,1), (1,1) ] int
        features     : torch.Tensor (1, ~108)   # history + scalars
    """
    
    p = state.current_player
    num_players = len(state.players_state)
    
    # Cards
    
    hole_cards = [ int(c.suit)*13 + int(c.rank) for c in state.players_state[p].hand ]
    board_cards = [ int(c.suit)*13 + int(c.rank) for c in state.public_cards ]
    
    hole_t = torch.tensor(hole_cards, dtype=torch.long, device=device).unsqueeze(0)  # (1, 2)

    flop_t  = torch.full((1, 3), -1, dtype=torch.long, device=device)
    turn_t  = torch.full((1, 1), -1, dtype=torch.long, device=device)
    river_t = torch.full((1, 1), -1, dtype=torch.long, device=device)
    
    n_board = len(board_cards)
    if n_board >= 3: flop_t[0]  = torch.tensor(board_cards[:3], dtype=torch.long, device=device)
    if n_board >= 4: turn_t[0,0] = board_cards[3]
    if n_board >= 5: river_t[0,0] = board_cards[4]
    
    cards_groups = (hole_t, flop_t, turn_t, river_t)
    
    # Scalars
    
    stacks = [ state.players_state[i].stake for i in range(num_players) ]
    hero_stack = stacks[p]
    opp_stack = stacks[1-p]
    
    eff_stack = min(hero_stack, opp_stack) if hero_stack > 0 else 100.0
    
    current_pot = state.pot
    
    street = 0 if len(board_cards) < 3 else len(board_cards) - 2
    
    button = state.button
    is_hero_button = 1 if p == state.button else 0
    
    # Bet history
    
    history_vec = torch.zeros((1, max_history_actions * 4), dtype=torch.float32, device=device)
    for i, (act_type, size_bb) in enumerate(action_history[:max_history_actions]):
        offset = i * 4
        history_vec[0, offset + 0] = 1.0                                      # occurred

        is_raise = 1.0 if act_type in {"raise", "allin", "bet"} else 0.0
        history_vec[0, offset + 1] = is_raise

        pot_before = pot_before_action[i] if i < len(pot_before_action) else current_pot
        rel_size = size_bb / max(pot_before, 0.01) if is_raise else 0.0
        rel_size = min(rel_size, 10.0)
        history_vec[0, offset + 2] = rel_size

        pot_frac = pot_before / 400.0
        history_vec[0, offset + 3] = min(pot_frac, 5.0)
        
    # Scalar features
    
    scalars = torch.zeros((1, 12), dtype=torch.float32, device=device)

    scalars[0, 0] = current_pot / 200.0
    scalars[0, 1] = hero_stack / 200.0
    scalars[0, 2] = opp_stack / 200.0
    scalars[0, 3] = eff_stack / 200.0

    if 0 <= street <= 3:
        scalars[0, 4 + street] = 1.0

    scalars[0, 8] = 1.0 if is_hero_button else 0.0   

    if action_history:
        last_act, last_size = action_history[-1]
        scalars[0, 9] = 1.0 if last_act in {"raise", "allin", "bet"} else 0.0
        if last_act in {"raise", "allin", "bet"}:
            last_pot = pot_before_action[-1] if pot_before_action else current_pot
            scalars[0, 10] = min(last_size / max(last_pot, 0.01), 10.0)
        scalars[0, 11] = len(action_history) / 20.0
    
    # Combine
    
    features = torch.cat([history_vec, scalars], dim=1)   # (1, 96 + 12 = 108)

    return cards_groups, features

def encode_state2(state: pkrs.State,
    action_history: list[tuple[str, float]],
    pot_before_action: list[float],
    max_history_actions: int=24,
    device: torch.device = torch.device('cpu'),
    history_template: torch.Tensor=None,
    scalars_template: torch.Tensor=None) -> tuple[list[torch.Tensor], torch.Tensor]:
    
    p = state.current_player
    num_players = len(state.players_state)

    # ─── Cards ────────────────────────────────────────────────────────
    # Avoid list comprehension + int() calls → direct access
    hand = state.players_state[p].hand
    hole_cards = torch.tensor(
        [int(c.suit) * 13 + int(c.rank) for c in hand],  # still needed, but only 2 elements
        dtype=torch.long, device=device
    ).unsqueeze(0)

    board = state.public_cards
    n_board = len(board)

    flop_t  = torch.full((1, 3), -1, dtype=torch.long, device=device)
    turn_t  = torch.full((1, 1), -1, dtype=torch.long, device=device)
    river_t = torch.full((1, 1), -1, dtype=torch.long, device=device)

    if n_board >= 3:
        flop_t[0, :min(3, n_board)] = torch.tensor(
            [int(c.suit) * 13 + int(c.rank) for c in board[:3]],
            dtype=torch.long, device=device
        )
    if n_board >= 4:
        turn_t[0, 0] = int(board[3].suit) * 13 + int(board[3].rank)
    if n_board >= 5:
        river_t[0, 0] = int(board[4].suit) * 13 + int(board[4].rank)

    cards_groups = (hole_cards, flop_t, turn_t, river_t)

    # ─── Scalars ───────────────────────────────────────────────────────
    stacks = [ps.stake for ps in state.players_state]
    hero_stack = stacks[p]
    opp_stack = stacks[1 - p] if num_players == 2 else 0.0
    eff_stack = min(hero_stack, opp_stack) if hero_stack > 0 else 100.0
    current_pot = state.pot

    street = 0 if n_board < 3 else n_board - 2
    is_hero_button = 1.0 if p == state.button else 0.0

    # ─── History vector ────────────────────────────────────────────────
    # Pre-allocate outside function if possible, or reuse
    if history_template is None:
        history_vec = torch.zeros((1, max_history_actions * 4), dtype=torch.float32, device=device)
    else:
        history_vec = history_template.clone().zero_()

    n_hist = min(len(action_history), max_history_actions)

    if n_hist > 0:
        # Vectorized version (much faster than for-loop)
        types, sizes = zip(*action_history[:n_hist])
        types = list(types)
        sizes = torch.tensor(sizes, dtype=torch.float32, device=device)

        is_raise = torch.tensor(
            [t in {"raise", "allin", "bet"} for t in types],
            dtype=torch.float32, device=device
        )

        # pot_before – use last value if missing
        pots = torch.tensor(
            [pot_before_action[i] if i < len(pot_before_action) else current_pot
             for i in range(n_hist)],
            dtype=torch.float32, device=device
        )

        rel_size = torch.where(
            is_raise.bool(),
            torch.minimum(sizes / pots.clamp(min=0.01), torch.tensor(10.0, device=device)),
            torch.zeros_like(sizes)
        )

        pot_frac = torch.minimum(pots / 400.0, torch.tensor(5.0, device=device))

        # Write in bulk
        indices = torch.arange(n_hist, device=device) * 4
        history_vec[0, indices]     = 1.0                # occurred
        history_vec[0, indices+1]   = is_raise
        history_vec[0, indices+2]   = rel_size
        history_vec[0, indices+3]   = pot_frac

    # ─── Scalars ───────────────────────────────────────────────────────
    if scalars_template is None:
        scalars = torch.zeros((1, 12), dtype=torch.float32, device=device)
    else:
        scalars = scalars_template.clone().zero_()

    scalars[0, 0] = current_pot / 200.0
    scalars[0, 1] = hero_stack / 200.0
    scalars[0, 2] = opp_stack / 200.0
    scalars[0, 3] = eff_stack / 200.0

    if 0 <= street <= 3:
        scalars[0, 4 + street] = 1.0

    scalars[0, 8] = is_hero_button

    if n_hist > 0:
        last_type, last_size = action_history[-1]
        scalars[0, 9] = 1.0 if last_type in {"raise", "allin", "bet"} else 0.0
        if scalars[0, 9] == 1.0:
            last_pot = pot_before_action[-1] if pot_before_action else current_pot
            scalars[0, 10] = min(last_size / max(last_pot, 0.01), 10.0)
        scalars[0, 11] = n_hist / 20.0

    features = torch.cat([history_vec, scalars], dim=1)

    return cards_groups, features