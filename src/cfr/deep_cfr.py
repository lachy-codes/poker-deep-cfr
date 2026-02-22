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
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import Dict, Optional

class DeepCFR:
    

    def deep_cfr(self, num_iterations, traversals, num_players:int=2, path: str="models", device: str="cpu"):
        
        
        self.HISTORY_TEMPLATE = torch.zeros((1, 24*4), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.SCALARS_TEMPLATE = torch.zeros((1, 12), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    
        plots_path = f"{path}/plots"
    
        os.makedirs(path, exist_ok=True)
        os.makedirs(plots_path, exist_ok=True)
        
        # Create CFR agents
        agents = [ DeepCFRAgent(num_actions=3) for _ in range(num_players) ]
        
        # Create strategy network, optimizer and memory
        self.strategy_net = DeepPokerNN(card_groups=4, bet_features=108, actions=3).to(device)
        self.strategy_optimizer = Adam(self.strategy_net.parameters(), lr=5e-5, weight_decay=1e-5)
        self.strat_mem = ReservoirMemory(3_000_000)
        
        self.num_actions=3
        
        avg_total_losses = [[] for _ in range(2)]
        avg_adv_losses = [[] for _ in range(2)]
        avg_sizing_losses = [[] for _ in range(2)]
        
        avg_strat_losses = []
        avg_strat_sizing_loss = []
        avg_strat_total_loss = []
        
        for iteration in range(1, num_iterations + 1):
            print(f"Iteration {iteration}/{num_iterations}")
            for p in range(num_players):
                
                print("\tCollecting data...")
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
                print("\tTraining Adv. Network...")
                total_loss = []
                adv_loss = []
                size_loss = []
                for step in range(100):
                    results = agents[p].train_advantage_network()

                    total_loss.append(results["total_loss"])
                    adv_loss.append(results["adv_loss"])
                    size_loss.append(results["size_loss"])
                
                avg_total_losses[p].append(sum(total_loss) / len(total_loss))
                avg_adv_losses[p].append(sum(adv_loss) / len(adv_loss))
                avg_sizing_losses[p].append(sum(size_loss) / len(size_loss))
                
                plt.plot(avg_total_losses[p])
                plt.xlabel("step")
                plt.ylabel("total_loss")
                plt.title(f"Total Losses iter {iteration} player {p}")
                plt.savefig(f"{plots_path}/total_iter_{iteration}_p_{p}.png", dpi=150)
                plt.close()
                
                plt.plot(avg_adv_losses[p])
                plt.xlabel("step")
                plt.ylabel("adv_loss")
                plt.title(f"Adv Losses iter {iteration} player {p}")
                plt.savefig(f"{plots_path}/adv_iter_{iteration}_p_{p}.png", dpi=150)
                plt.close()
                
                plt.plot(avg_sizing_losses[p])
                plt.xlabel("step")
                plt.ylabel("size_loss")
                plt.title(f"size Losses iter {iteration} player {p}")
                plt.savefig(f"{plots_path}/size_iter_{iteration}_p_{p}.png", dpi=150)
                plt.close()
                
            if iteration % 5 == 0 and iteration != 0:
                print("\ttraining strategy network...")
                metrics = self.train_strategy_network()
                avg_strat_losses.append(metrics["stategy_loss"])
                avg_strat_sizing_loss.append(metrics["sizing_loss"])
                avg_strat_total_loss.append(metrics["total_loss"])
                
                plt.plot(avg_strat_losses)
                plt.xlabel("step")
                plt.ylabel("strategy loss")
                plt.title(f"Strategy Loss iter {iteration}")
                plt.savefig(f"{plots_path}/strategy_{iteration}.png", dpi=150)
                
                plt.plot(avg_strat_sizing_loss)
                plt.xlabel("step")
                plt.ylabel("sizing loss")
                plt.title(f"Strategy Loss iter {iteration}")
                plt.savefig(f"{plots_path}/strategy_sizing_{iteration}.png", dpi=150)
                
                plt.plot(avg_strat_total_loss)
                plt.xlabel("step")
                plt.ylabel("strategy total loss")
                plt.title(f"Strategy total Loss iter {iteration}")
                plt.savefig(f"{plots_path}/strategy_total_{iteration}.png", dpi=150)
                
            if iteration % 100 == 0 and iteration != 0:
                os.makedirs(f"{path}/checkpoints")
                torch.save(self.strategy_net.state_dict(), f"{path}/checkpoints/checkpoint_{iteration}.pt")
    
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
            bet_size = self.noisify_bet_frac(state, iteration, bet_size)
            
            a_vals = np.zeros(self.num_actions)
            for a in legal_actions:
                action, act_type = self.action_value_to_action_obj(state, iteration, a, bet_size)
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
                weighted_regret = max(-10.0, min(10.0, weighted_regret))
                
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
            action, action_type = self.action_value_to_action_obj(state, iteration, action_int, bet_size)
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
    
    def action_value_to_action_obj(self, state: pkrs.State, iteration, action_int, bet_frac) -> tuple[pkrs.Action, str]:
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
    
    def noisify_bet_frac(self, state: pkrs.State, iteration: int, bet_frac: float): 
        # Add exploration noise
        temperature = max(0.5, 0.4 * (1 - iteration / 1000))
        noise = torch.randn(1).item() * temperature
        
        sampled_frac = bet_frac + noise
        sampled_frac = max(0.0, min(1.0 - 1e-2, sampled_frac))
        
        return sampled_frac

    def train_strategy_network(
        self,
        batch_size: int = 2048,
        epochs: int = 3,
        sizing_loss_weight: float = 0.08,          # tune between 0.05–0.15
        grad_clip_max_norm: float = 1.0,
        use_amp: bool = True,
        scaler: Optional[GradScaler] = None,
    ) -> Dict[str, float]:
        """
        Trains the Strategy Network in Classic Deep CFR.
        
        Sample tuple from strategy_memory:
            (card_groups, features, bet_size_if_raise_else_0, target_strategy, iteration_weight)
        
        - target_strategy: torch.Tensor (batch, n_actions) — target probability distribution
        - bet_size_if_raise_else_0: normalized [0,1] if raise was chosen, else 0.0
        - iteration_weight: scalar (usually iteration number)
        """
        if self.strat_mem.length < batch_size:
            return {"strategy_loss": 0.0, "sizing_loss": 0.0, "total_loss": 0.0}

        self.strategy_net.train()

        total_ce_loss = 0.0
        total_size_loss = 0.0
        total_steps = 0

        for epoch in range(epochs):
            # Uniform sampling from reservoir
            batch = self.strat_mem.sample_batch(batch_size)

            # Collate
            card_groups_list = [[] for _ in range(4)]  # hole, flop, turn, river
            features_list = []
            target_strategy_list = []
            bet_size_list = []
            weight_list = []

            for cg, feat, bet_size, target_strat, weight in batch:
                for i, g in enumerate(cg):
                    card_groups_list[i].append(g)
                features_list.append(feat)
                target_strategy_list.append(target_strat)
                bet_size_list.append(bet_size)
                weight_list.append(weight)

            # Stack on device
            card_groups_batch = [torch.cat(lst, dim=0).to(self.device) for lst in card_groups_list]
            features_batch = torch.cat(features_list, dim=0).to(self.device)

            target_strategy_batch = torch.stack(target_strategy_list, dim=0).to(self.device)
            bet_size_batch = torch.tensor(bet_size_list, dtype=torch.float32, device=self.device)
            weight_batch = torch.tensor(weight_list, dtype=torch.float32, device=self.device)

            # Forward pass
            self.optimizer_strategy.zero_grad(set_to_none=True)

            if use_amp and scaler is not None:
                with autocast(device_type=self.device.type):
                    strategy_logits, pred_bet_frac = self.strategy_net(card_groups_batch, features_batch)

                    # 1. Strategy loss: explicit softmax → log_softmax → cross-entropy with soft targets
                    log_probs = F.log_softmax(strategy_logits, dim=-1)
                    ce_loss_per_sample = - (target_strategy_batch * log_probs).sum(dim=-1)
                    weighted_ce_loss = (ce_loss_per_sample * weight_batch).mean()

                    # 2. Bet sizing loss: Smooth L1 (Huber) — only on raise actions
                    size_loss = torch.tensor(0.0, device=self.device)
                    is_raise = (bet_size_batch > 0.0)   # your convention: >0 means raise

                    if is_raise.any():
                        pred_frac = pred_bet_frac.squeeze(-1)[is_raise]
                        true_frac = bet_size_batch[is_raise]
                        size_loss = F.smooth_l1_loss(
                            pred_frac,
                            true_frac,
                            reduction='mean',
                            beta=0.1                     # smooth transition zone
                        )

                    total_loss = weighted_ce_loss + sizing_loss_weight * size_loss

                scaler.scale(total_loss).backward()
                scaler.unscale_(self.optimizer_strategy)
                torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), grad_clip_max_norm)
                scaler.step(self.optimizer_strategy)
                scaler.update()

            else:
                # No AMP fallback
                strategy_logits, pred_bet_frac = self.strategy_net(card_groups_batch, features_batch)

                log_probs = F.log_softmax(strategy_logits, dim=-1)
                ce_loss_per_sample = - (target_strategy_batch * log_probs).sum(dim=-1)
                weighted_ce_loss = (ce_loss_per_sample * weight_batch).mean()

                size_loss = torch.tensor(0.0, device=self.device)
                is_raise = (bet_size_batch > 0.0)
                if is_raise.any():
                    pred_frac = pred_bet_frac.squeeze(-1)[is_raise]
                    true_frac = bet_size_batch[is_raise]
                    size_loss = F.smooth_l1_loss(
                        pred_frac,
                        true_frac,
                        reduction='mean',
                        beta=0.1
                    )

                total_loss = weighted_ce_loss + sizing_loss_weight * size_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), grad_clip_max_norm)
                self.optimizer_strategy.step()

            total_ce_loss += weighted_ce_loss.item()
            total_size_loss += size_loss.item()
            total_steps += 1

        avg_ce_loss = total_ce_loss / total_steps
        avg_size_loss = total_size_loss / total_steps

        return {
            "strategy_loss": avg_ce_loss,           # cross-entropy on policy distribution
            "sizing_loss":   avg_size_loss,         # smooth L1 on bet fraction
            "total_loss":    avg_ce_loss + sizing_loss_weight * avg_size_loss,
            "steps":         total_steps,
            "batch_size":    batch_size,
            "epochs":        epochs,
        }