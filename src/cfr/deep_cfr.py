from src.memory.reservoir_memory import ReservoirMemory
import pokers as pkrs
from src.agents.deep_cfr_agent import DeepCFRAgent
from src.networks.deep_poker_network import DeepPokerNN
from src.encoding.encode import encode_state2
from torch.optim import Adam
import numpy as np
import os
import time
import random
import cProfile
import pstats
import torch

class DeepCFR:
    

    def deep_cfr(self, num_iterations, traversals, num_players:int=2, path: str="models", device: str="cpu"):
        
        
        self.HISTORY_TEMPLATE = torch.zeros((1, 24*4), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.SCALARS_TEMPLATE = torch.zeros((1, 12), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    
        os.makedirs(path, exist_ok=True)
        
        # Create CFR agents
        agents = [ DeepCFRAgent(num_actions=3) for _ in range(num_players) ]
        
        # Create strategy network, optimizer and memory
        strategy_net = DeepPokerNN(card_groups=4, bet_features=108, actions=3).to(device)
        strategy_optimizer = Adam(strategy_net.parameters(), lr=5e-5, weight_decay=1e-5)
        self.strat_mem = ReservoirMemory()
        
        self.num_actions=3
        
        for iteration in range(1, num_iterations + 1):
            print(f"Iteration {iteration}/{num_iterations}")
            for p in range(num_players):
                
                
                for _ in range(traversals):
                    state = pkrs.State.from_seed(
                        n_players=num_players,
                        button=random.randint(0, num_players-1),
                        sb=1,
                        bb=2,
                        stake=200.0,
                        seed=random.randint(0, 2**32 - 1)
                    )
                    
                    self.cfr_traverse(state=state, iteration=iteration, traverser=p, agents=agents, pot_before_action=[], action_history=[])
                
    
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
            
            card_groups, features = encode_state2(state=state, action_history=action_history, pot_before_action=pot_before_action, history_template=self.HISTORY_TEMPLATE, scalars_template=self.SCALARS_TEMPLATE)
            
            # Get regret matched strategy
            strategy, bet_size = agents[traverser].regret_matched_strategy(card_groups, features, legal_actions)
            
            a_vals = np.zeros(self.num_actions)
            for a in legal_actions:
                action, act_type = self.action_value_to_action_obj(state, a, bet_size)
                new_state = state.apply_action(action)
                
                action_history.append((act_type, action.amount))
                pot_before_action.append(state.pot)
                
                if new_state.status != pkrs.StateStatus.Ok:
                    print(f"Invalid action - {a} {new_state.status}")
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
                    (card_groups, features, a, bet_size if 2 in legal_actions else 0.0, weighted_regret)
                )
            
            return ev
        else: 
            legal_actions = self.get_legal_actions(state=state, player=current_player)
            
            card_groups, features = encode_state2(state=state, action_history=action_history, pot_before_action=pot_before_action, history_template=self.HISTORY_TEMPLATE, scalars_template=self.SCALARS_TEMPLATE)
            
            strategy, bet_size = agents[current_player].regret_matched_strategy(card_groups, features, legal_actions)
            
            self.strat_mem.add(
                (card_groups, features, bet_size if 2 in legal_actions else 0.0, strategy, iteration)
            )
            
            action_int = np.random.choice(len(strategy), p=strategy)
            action, action_type = self.action_value_to_action_obj(state, action_int, bet_size)
            new_state = state.apply_action(action)
            
            if new_state.status != pkrs.StateStatus.Ok:
                print(f"Invalid action - {action_int} {new_state.status}...")
                return 0.0
            
            action_history.append((action_type, action.amount))
            pot_before_action.append(state.pot)
            
            return self.cfr_traverse(new_state, iteration, traverser, agents, pot_before_action, action_history)
            
            
    def get_legal_actions(self, state: pkrs.State, player: int) -> list[int]:
        pkrs_legal_actions = state.legal_actions
        legal_actions = []
        
        if pkrs.ActionEnum.Fold in pkrs_legal_actions:
            legal_actions.append(0)
        if pkrs.ActionEnum.Check in pkrs_legal_actions or pkrs.ActionEnum.Call in pkrs_legal_actions:
            legal_actions.append(1)
        if pkrs.ActionEnum.Raise in pkrs_legal_actions and self.can_raise(state):
            legal_actions.append(2)
                
        return legal_actions
    
    def action_value_to_action_obj(self, state: pkrs.State, action_int, bet_frac) -> tuple[pkrs.Action, str]:
        match action_int:
            case 0:
                return pkrs.Action(pkrs.ActionEnum.Fold), "fold"
            case 1:
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check), "check"
                else: 
                    return pkrs.Action(pkrs.ActionEnum.Call), "call"
            case 2: 
                current_bet = state.players_state[state.current_player].bet_chips
                stake = state.players_state[state.current_player].stake
                to_call = max(0.0, state.min_bet - current_bet)
                min_raise = 1
                max_raise = stake - to_call
                
                amount = min_raise + bet_frac * (max_raise - min_raise)
                
                if state.apply_action(
                    pkrs.Action(pkrs.ActionEnum.Raise, amount)
                ).status != pkrs.StateStatus.Ok:
                    print(f"current_bet: {current_bet}")
                    print(f"stake: {stake}")
                    print(f"min_bet: {state.min_bet}")
                    print(f"to call: {to_call}")
                    print(f"max raise: {max_raise}")
                    print(f"min raise: {min_raise}")
                    print(f"amount: {amount}")
                
                
                return pkrs.Action(pkrs.ActionEnum.Raise, amount), "raise"
            case _:
                print("Invalid action passed. Folding...")
                return pkrs.Action(pkrs.ActionEnum.Fold)
            
    def can_raise(self, state: pkrs.State):
        current_bet = state.players_state[state.current_player].bet_chips
        stake = state.players_state[state.current_player].stake
        
        to_call = max(0.0, state.min_bet - current_bet)
        max_raise = stake - to_call
        if max_raise <= 1:
            return False
        else:
            return True