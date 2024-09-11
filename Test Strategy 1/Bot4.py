from Strategy import StrategyBase
import numpy as np

class UserStrategy(StrategyBase):

    def __init__(self):
        # Initialize tracking variables
        self.bid_history = []
        self.round_number = 0
        self.capital_history = []
        self.mean_highest_bid = 0
        self.mean_second_highest_bid = 0
        self.std_highest_bid = 0
        self.std_second_highest_bid = 0
        self.alpha = 0.1  # Learning rate for Bayesian updating
        self.safety_margin = 0.1  # Margin for ensuring positive payoff

    def update_statistics(self, previous_winners, previous_second_highest_bids):
        '''
        Update statistics such as mean and standard deviation based on historical data
        to better predict opponents' bidding behavior.
        '''
        if previous_winners:
            self.mean_highest_bid = np.mean(previous_winners)
            self.std_highest_bid = np.std(previous_winners)
        if previous_second_highest_bids:
            self.mean_second_highest_bid = np.mean(previous_second_highest_bids)
            self.std_second_highest_bid = np.std(previous_second_highest_bids)

    def calculate_bid(self, current_value, predicted_highest, predicted_second_highest, capital):
        '''
        Calculate bid using Bayesian estimation and risk adjustment.
        '''
        # Bayesian updating of estimates
        estimated_highest = self.alpha * predicted_highest + (1 - self.alpha) * self.mean_highest_bid
        estimated_second_highest = self.alpha * predicted_second_highest + (1 - self.alpha) * self.mean_second_highest_bid

        # Calculate safe bid based on estimated highest bid
        bid = estimated_second_highest + (estimated_highest - estimated_second_highest) * 0.5

        # Ensure the bid is within capital and value constraints
        bid = min(bid, capital, current_value - self.safety_margin)
        
        # Prevent bidding too low
        bid = max(bid, 0.1)
        
        return bid

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Advanced Strategy for Variation 1:
        - Use Bayesian estimation to predict opponents' bids.
        - Adjust bids based on current capital and historical data.
        - Ensure positive payoff and manage risk.
        '''
        self.round_number += 1
        
        # Update statistics based on previous rounds
        self.update_statistics(previous_winners, previous_second_highest_bids)
        
        # Predict opponent behavior based on historical data
        predicted_highest = self.mean_highest_bid if previous_winners else current_value
        predicted_second_highest = self.mean_second_highest_bid if previous_second_highest_bids else current_value * 0.5
        
        # Calculate the bid
        bid = self.calculate_bid(current_value, predicted_highest, predicted_second_highest, capital)
        
        # Track the bid history
        self.bid_history.append(bid)
        self.capital_history.append(capital)

        return bid
