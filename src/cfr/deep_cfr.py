from memory.reservoir_memory import ReservoirMemory
import pokers as pkrs
from agents.deep_cfr_agent import DeepCFRAgent
from networks.deep_poker_network import DeepPokerNN
from encoding.encode import encode_state
from torch.optim import Adam
import numpy as np
import os
import time
import random

class DeepCFR:
    def deep_cfr(self, num_iterations, traversals, num_players:int=2, path: str="models", device: str="cpu"):
        
        os.makedirs(path, exist_ok=True)
        
        # Create CFR agents
        agents = [ DeepCFRAgent() for _ in range(num_players) ]
        
        # Create strategy network, optimizer and memory
        strategy_net = DeepPokerNN(card_groups=4, bet_features=108, actions=10, hidden_dim=256).to(device)
        strategy_optimizer = Adam(strategy_net.parameters(), lr=5e-5, weight_decay=1e-5)
        self.strat_mem = ReservoirMemory()
        
        self.num_actions=10
        
        for iteration in range(1, num_iterations + 1):
            print(f"Iteration {iteration}/{num_iterations}")
            for p in range(num_players):
                start_time = time.time()
                for _ in range(traversals):
                    state = pkrs.State.from_seed(
                        n_players=num_players,
                        button=random.randint(0, num_players-1),
                        sb=1,
                        bb=2,
                        stake=200.0,
                        seed=random.randint(0, 2**32 - 1)
                    )
                    
                    self.cfr_traverse(state, iteration, p, agents, [], [])
                end_time = time.time()
                print(f"time taken: {(end_time - start_time)}")
    
    def cfr_traverse(self, 
                    state: pkrs.State, 
                    iteration: int, 
                    traverser: int, 
                    agents: list[DeepCFRAgent],
                    pot_before_action: list[float],
                    action_history: list[tuple[str, float]], 
                    device: str="cpu") -> float: 
        """
        Recursive external-sampling MCCFR traversal.
        
        Returns: counterfactual value for the traverser at this node.
        
        Side effects:
        - Appends to action_history & pot_before_action during recursion
        - Adds samples to advantage_buffer and strategy_buffer
        """
    
        if state.final_state:
            return state.players_state[traverser].reward
        
        current_player = state.current_player
        
        if current_player == traverser:
            legal_actions = self.get_legal_actions(state, current_player)
            
            card_groups, features = encode_state(state=state, action_history=action_history, pot_before_action=pot_before_action)
            
            # Get regret matched strategy
            strategy = agents[traverser].regret_matched_strategy(card_groups, features, legal_actions)
            
            a_vals = np.zeros(self.num_actions)
            for a in legal_actions:
                action, act_type = self.action_value_to_action_obj(state, a)
                new_state = state.apply_action(action)
                
                action_history.append((act_type, action.amount))
                pot_before_action.append(state.pot)
                
                if new_state.status != pkrs.StateStatus.Ok:
                    print("Invalid action - Skipping...")
                    continue
                
                a_vals[a] = self.cfr_traverse(
                    new_state, 
                    iteration, 
                    traverser,
                    agents, 
                    pot_before_action, 
                    action_history
                )
                
                action_history.pop()
                pot_before_action.pop()
            
            ev = sum(strategy[a] * a_vals[a] for a in legal_actions)
            
            norm_factor = max(abs(max(a_vals)), abs(min(a_vals)), 1.0)
            for a in legal_actions:
                regret = a_vals[a] - ev
                
                # normalize and clip regrets
                regret_norm = regret / norm_factor
                clipped_regret = np.clip(regret, -10, 10)
                
                # weight regret
                weight = np.sqrt(iteration) if iteration > 1 else 1.0
                weighted_regret = clipped_regret * weight
                
                agents[traverser].memory.add(
                    (card_groups, features, a, weighted_regret)
                )
            
            return ev
        else: 
            legal_actions = self.get_legal_actions(state, current_player)
            
            card_groups, features = encode_state(state=state, action_history=action_history, pot_before_action=pot_before_action)
            
            strategy = agents[current_player].regret_matched_strategy(card_groups, features, legal_actions)
            
            self.strat_mem.add(
                (card_groups, features, strategy, iteration)
            )
            
            action_int = np.random.choice(len(strategy), p=strategy)
            action, action_type = self.action_value_to_action_obj(state, action_int)
            new_state = state.apply_action(action)
            
            if new_state.status != pkrs.StateStatus.Ok:
                print("Invalid action - (Returning 0)...")
                return 0.0
            
            action_history.append((act_type, action.amount))
            pot_before_action.append(state.pot)
            
            return self.cfr_traverse(new_state, iteration, traverser, agents, pot_before_action, action_history)
            
            
    def get_legal_actions(state: pkrs.State, player: int) -> list[int]:
        pkrs_legal_actions = state.legal_actions
        legal_actions = []
        
        if pkrs.ActionEnum.Fold in pkrs_legal_actions:
            legal_actions.append(0)
        if pkrs.ActionEnum.Check in pkrs_legal_actions or pkrs.ActionEnum.Call in pkrs_legal_actions:
            legal_actions.append(1)
        if pkrs.ActionEnum.Raise in pkrs_legal_actions:
            legal_actions.append(2)
            
            stake = state.players_state[player].stake
            bet_chips = state.players_state[player].bet_chips
            
            max_raise = stake - state.min_bet + bet_chips
            
            if state.pot // 2 <= max_raise:
                legal_actions.append(3)
            if 3 * state.pot // 4 <= max_raise:
                legal_actions.append(4)
            if state.pot <= max_raise:
                legal_actions.append(5)
            if 3 * state.pot // 2 <= max_raise:
                legal_actions.append(6)
            if 2 * state.pot <= max_raise:
                legal_actions.append(7)
            if 3 * state.pot <= max_raise:
                legal_actions.append(8)
            if 3 * state.pot < max_raise:
                legal_actions.append(9)
                
        return legal_actions
    
    def action_value_to_action_obj(self, state: pkrs.State, action_int) -> tuple[pkrs.Action, str]:
        match action_int:
            case 0:
                return pkrs.Action(pkrs.ActionEnum.Fold), "fold"
            case 1:
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check), "check"
                else: 
                    return pkrs.Action(pkrs.ActionEnum.Call), "call"
            case 2: 
                return pkrs.Action(pkrs.ActionEnum.Raise, state.min_bet), "raise"
            case 3:
                return pkrs.Action(pkrs.ActionEnum.Raise, state.pot // 2), "raise"
            case 4: 
                return pkrs.Action(pkrs.ActionEnum.Raise, 3 * state.pot // 4), "raise"
            case 5:
                return pkrs.Action(pkrs.ActionEnum.Raise, state.pot), "raise"
            case 6:
                return pkrs.Action(pkrs.ActionEnum.Raise, 3 * state.pot // 2), "raise"
            case 7:
                return pkrs.Action(pkrs.ActionEnum.Raise, 2 * state.pot), "raise"
            case 8:
                return pkrs.Action(pkrs.ActionEnum.Raise, 3 * state.pot), "raise"
            case 9:
                return pkrs.Action(pkrs.ActionEnum.Raise, 
                    state.players_state[state.current_player].stake + state.players_state[state.current_player].bet_chips - state.min_bet), "allin"
            case _:
                return pkrs.Action(pkrs.ActionEnum.Fold), "fold"