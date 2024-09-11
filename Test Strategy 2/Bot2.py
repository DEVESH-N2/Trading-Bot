from Strategy import StrategyBase
import numpy as np

class AdvancedUserStrategy(StrategyBase):
    
    def __init__(self):
        self.round_number = 0
        self.alpha = 0.2  # Learning rate for Bayesian updating
        self.safety_margin = 0.1
        self.aggression_factor = 0.5  # Starting aggression factor
        self.opponent_bid_history = []
        self.capital_history = []
        self.mean_highest_bid = 0
        self.std_highest_bid = 1  # Default to avoid zero division errors

    def update_statistics(self, previous_winners):
        '''
        Update estimation of opponents' bidding behavior using Bayesian updating.
        '''
        if previous_winners:
            self.mean_highest_bid = np.mean(previous_winners)
            self.std_highest_bid = np.std(previous_winners)

    def estimate_opponent_bids(self):
        '''
        Estimate the maximum bid likely to be made by opponents based on historical data.
        '''
        predicted_max_bid = self.mean_highest_bid + self.alpha * self.std_highest_bid
        return predicted_max_bid

    def calculate_optimal_bid(self, current_value, predicted_max_bid, capital):
        '''
        Calculate the optimal bid by balancing the potential maximum value and risk of overbidding.
        '''
        bid = predicted_max_bid * self.aggression_factor
        bid = min(bid, capital, current_value - self.safety_margin)
        return max(bid, 0.1)  # Ensure bid is not too low

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Refactored strategy to match the required template.
        '''
        self.round_number += 1

        # Update opponent statistics
        self.update_statistics(previous_winners)
        
        # Estimate the maximum bid opponents will likely place
        predicted_max_bid = self.estimate_opponent_bids()
        
        # Calculate optimal bid
        bid = self.calculate_optimal_bid(current_value, predicted_max_bid, capital)
        
        # Track bid and capital history for analysis
        self.opponent_bid_history.append(predicted_max_bid)
        self.capital_history.append(capital)

        return bid
