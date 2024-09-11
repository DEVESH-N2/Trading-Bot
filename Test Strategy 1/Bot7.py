from Strategy import StrategyBase
import numpy as np

class UserStrategy(StrategyBase):

    def __init__(self):
        # Initialize tracking variables
        self.round_number = 0
        self.capital_history = []
        self.bid_history = []
        self.highest_bid_history = []
        self.second_highest_bid_history = []
        self.alpha = 0.5  # Learning rate for Bayesian updating
        self.safety_margin = 0.05  # Margin to ensure positive payoff
        self.bid_scaling_factor = 0.7  # Factor to scale bids based on predictions
        self.capital_threshold_high = 100  # High capital threshold for aggression
        self.capital_threshold_low = 30   # Low capital threshold for conservatism
        self.round_threshold = 500        # Threshold for adjusting aggression based on rounds
        self.capital_buffer = 0.1         # Buffer to avoid running out of capital
        self.prediction_weight = 0.8      # Weight to balance between prediction and current value

    def update_statistics(self, previous_winners, previous_second_highest_bids):
        '''
        Update statistics based on historical data
        '''
        if previous_winners:
            self.mean_highest_bid = np.mean(previous_winners)
            self.std_highest_bid = np.std(previous_winners)
        else:
            self.mean_highest_bid = 0
            self.std_highest_bid = 1  # Prevent division by zero

        if previous_second_highest_bids:
            self.mean_second_highest_bid = np.mean(previous_second_highest_bids)
            self.std_second_highest_bid = np.std(previous_second_highest_bids)
        else:
            self.mean_second_highest_bid = 0
            self.std_second_highest_bid = 1  # Prevent division by zero

    def estimate_opponent_bids(self, previous_winners, previous_second_highest_bids):
        '''
        Estimate opponent bids using Bayesian updating
        '''
        if previous_winners:
            predicted_highest = self.mean_highest_bid + self.alpha * self.std_highest_bid
        else:
            predicted_highest = self.mean_highest_bid

        if previous_second_highest_bids:
            predicted_second_highest = self.mean_second_highest_bid + self.alpha * self.std_second_highest_bid
        else:
            predicted_second_highest = self.mean_second_highest_bid

        return predicted_highest, predicted_second_highest

    def calculate_optimal_bid(self, current_value, predicted_highest, predicted_second_highest, capital):
        '''
        Calculate the optimal bid balancing risk and expected payoff
        '''
        # Predict the bid as a combination of current value and predicted opponent behavior
        predicted_bid = predicted_second_highest + (predicted_highest - predicted_second_highest) * self.bid_scaling_factor
        bid = self.prediction_weight * predicted_bid + (1 - self.prediction_weight) * current_value

        # Ensure bid is within constraints
        bid = min(bid, capital - self.capital_buffer, current_value - self.safety_margin)
        bid = max(bid, 0.1)  # Avoid bids that are too low

        return bid

    def adjust_aggression(self, capital, round_number):
        '''
        Adjust aggression based on capital and round number
        '''
        if capital > self.capital_threshold_high and round_number < self.round_threshold:
            self.alpha = min(1.0, self.alpha + 0.1)  # Increase aggression
        elif capital < self.capital_threshold_low or round_number > self.round_threshold:
            self.alpha = max(0.2, self.alpha - 0.1)  # Decrease aggression

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Main function to decide the bid amount
        '''
        self.round_number += 1
        
        # Update statistics with previous rounds' data
        self.update_statistics(previous_winners, previous_second_highest_bids)
        
        # Estimate opponent bids
        predicted_highest, predicted_second_highest = self.estimate_opponent_bids(previous_winners, previous_second_highest_bids)
        
        # Calculate optimal bid
        bid = self.calculate_optimal_bid(current_value, predicted_highest, predicted_second_highest, capital)
        
        # Adjust aggression level based on game progress
        self.adjust_aggression(capital, self.round_number)
        
        # Track bid and capital history
        self.bid_history.append(bid)
        self.capital_history.append(capital)

        return bid
