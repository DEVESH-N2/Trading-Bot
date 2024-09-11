from Strategy import StrategyBase
import numpy as np
from scipy.stats import norm

class HybridBayesianRLStrategy(StrategyBase):
    
    def __init__(self):
        self.round_number = 0
        self.alpha = 0.1  # Smoothing factor for Bayesian updating
        self.safety_margin = 0.1
        self.initial_aggression_factor = 0.5
        self.max_aggression_factor = 0.9
        self.min_aggression_factor = 0.2
        self.aggression_factor = self.initial_aggression_factor
        self.history_winners = []
        self.history_second_highest = []
        self.capital_history = []
        self.num_bidders_history = []
        self.q_values = {}  # Q-values for reinforcement learning
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_prob = 0.1

    def update_statistics(self, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Bayesian updating of opponent behavior estimates.
        '''
        self.history_winners.extend(previous_winners)
        self.history_second_highest.extend(previous_second_highest_bids)
        self.capital_history.append(capital)
        self.num_bidders_history.append(num_bidders)
        
        # Estimate parameters for Bayesian updating
        if self.history_winners:
            self.mean_winner = np.mean(self.history_winners)
            self.std_winner = np.std(self.history_winners)
            self.mean_second_highest = np.mean(self.history_second_highest)
            self.std_second_highest = np.std(self.history_second_highest)
        else:
            self.mean_winner = 0
            self.std_winner = 1
            self.mean_second_highest = 0
            self.std_second_highest = 1

    def estimate_opponent_bids(self, capital, num_bidders):
        '''
        Estimate opponent bids using Bayesian inference.
        '''
        mean_highest = self.mean_winner + self.alpha * (capital / max(capital, 1))
        std_highest = self.std_winner
        mean_second_highest = self.mean_second_highest + self.alpha * (num_bidders / max(num_bidders, 1))
        std_second_highest = self.std_second_highest
        
        # Predict the highest and second-highest bids
        predicted_max_bid = norm.rvs(loc=mean_highest, scale=std_highest)
        predicted_second_highest_bid = norm.rvs(loc=mean_second_highest, scale=std_second_highest)
        
        return predicted_max_bid, predicted_second_highest_bid

    def adjust_aggression(self, capital):
        '''
        Adjust aggression based on remaining capital.
        '''
        capital_factor = min(1, capital / 100)
        self.aggression_factor = self.min_aggression_factor + (self.max_aggression_factor - self.min_aggression_factor) * capital_factor

    def calculate_optimal_bid(self, current_value, predicted_max_bid, predicted_second_highest_bid, capital):
        '''
        Calculate the optimal bid.
        '''
        base_bid = predicted_second_highest_bid + (predicted_max_bid - predicted_second_highest_bid) * self.aggression_factor
        adjusted_bid = base_bid * (current_value / max(current_value, 1))  # Normalize based on current value
        adjusted_bid = min(adjusted_bid, capital, current_value - self.safety_margin)
        
        return max(adjusted_bid, 0.1)  # Ensure bid is not too low

    def q_learning_update(self, state, action, reward, next_state):
        '''
        Update Q-values using the Q-learning algorithm.
        '''
        if state not in self.q_values:
            self.q_values[state] = {}
        if action not in self.q_values[state]:
            self.q_values[state][action] = 0

        current_q = self.q_values[state][action]
        max_next_q = max(self.q_values.get(next_state, {}).values(), default=0)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_values[state][action] = new_q

    def choose_action(self, state, available_actions):
        '''
        Choose an action based on the exploration-exploitation trade-off.
        '''
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(available_actions)
        else:
            q_values = [self.q_values.get(state, {}).get(a, 0) for a in available_actions]
            max_q = max(q_values, default=0)
            return np.random.choice([a for a, q in zip(available_actions, q_values) if q == max_q])

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Main method to determine the bid.
        '''
        self.round_number += 1

        # Update statistics with historical data
        self.update_statistics(previous_winners, previous_second_highest_bids, capital, num_bidders)
        
        # Estimate opponent behavior
        predicted_max_bid, predicted_second_highest_bid = self.estimate_opponent_bids(capital, num_bidders)
        
        # Adjust aggression based on capital
        self.adjust_aggression(capital)
        
        # Define state and action
        state = (capital, num_bidders)
        available_actions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Example actions
        action = self.choose_action(state, available_actions)
        
        # Calculate the optimal bid
        bid = self.calculate_optimal_bid(current_value, predicted_max_bid, predicted_second_highest_bid, capital)
        
        # Simulate the result and update Q-values
        next_state = (capital - bid, num_bidders)  # Example next state
        reward = 0  # Calculate reward based on auction results
        self.q_learning_update(state, action, reward, next_state)
        
        return bid
