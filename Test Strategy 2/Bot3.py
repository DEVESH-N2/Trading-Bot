from Strategy import StrategyBase
import numpy as np

class AdvancedUserStrategy(StrategyBase):
    
    def __init__(self):
        self.round_number = 0
        self.alpha = 0.2  # Learning rate for Bayesian updating
        self.safety_margin = 0.1
        self.aggression_factor = 0.5  # Starting aggression factor
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
        # Adjust bid based on predicted maximum bid and aggression factor
        bid = predicted_max_bid * self.aggression_factor
        
        # Ensure bid is within constraints and adjust for safety margin
        bid = min(bid, capital, current_value - self.safety_margin)
        return max(bid, 0.1)  # Ensure bid is not too low

    def adjust_aggression(self, capital, num_bidders):
        '''
        Dynamically adjust aggression based on remaining capital and number of bidders.
        '''
        # Increase aggression when capital is high or if fewer bidders remain
        if capital > 100 or num_bidders < 5:
            self.aggression_factor = min(0.8, self.aggression_factor + 0.1)
        # Decrease aggression when capital is low or if many bidders are competing
        elif capital < 50 or num_bidders > 10:
            self.aggression_factor = max(0.3, self.aggression_factor - 0.1)

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Make a bid based on the strategy.
        '''
        self.round_number += 1

        # Update opponent statistics based on previous winners
        self.update_statistics(previous_winners)
        
        # Estimate the maximum bid opponents will likely place
        predicted_max_bid = self.estimate_opponent_bids()
        
        # Adjust aggression based on remaining capital and number of bidders
        self.adjust_aggression(capital, num_bidders)
        
        # Calculate optimal bid
        bid = self.calculate_optimal_bid(current_value, predicted_max_bid, capital)

        return bid
