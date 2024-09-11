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
        self.alpha = 0.2  # Learning rate for Bayesian updating
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

    def estimate_opponent_bids(self, previous_winners, previous_second_highest_bids):
        '''
        Estimate the likely behavior of opponents based on historical data.
        '''
        if not previous_winners:
            return 0, 0
        
        # Bayesian estimation of opponent behavior
        predicted_highest = self.mean_highest_bid if previous_winners else 0
        predicted_second_highest = self.mean_second_highest_bid if previous_second_highest_bids else 0
        
        return predicted_highest, predicted_second_highest

    def calculate_optimal_bid(self, current_value, predicted_highest, predicted_second_highest, capital):
        '''
        Calculate the optimal bid by balancing risk and expected payoff.
        '''
        # Estimate bid based on opponent behavior
        bid = predicted_second_highest + (predicted_highest - predicted_second_highest) * 0.6
        
        # Ensure bid is within constraints
        bid = min(bid, capital, current_value - self.safety_margin)
        
        # Ensure bid is not too low
        bid = max(bid, 0.1)
        
        return bid

    def adjust_aggression(self, capital, round_number):
        '''
        Adjust aggression level based on capital and round number.
        '''
        # Increase aggression if capital is high and we're in early rounds
        if capital > 100 and round_number < 500:
            self.alpha = min(0.9, self.alpha + 0.05)
        # Decrease aggression if capital is low or we're in late rounds
        elif capital < 30 or round_number > 800:
            self.alpha = max(0.1, self.alpha - 0.05)

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Advanced Strategy for Variation 1:
        - Use Bayesian estimation to predict opponents' bids.
        - Calculate optimal bid based on expected payoff.
        - Adjust bidding strategy based on current capital and round number.
        '''
        self.round_number += 1
        
        # Update statistics based on previous rounds
        self.update_statistics(previous_winners, previous_second_highest_bids)
        
        # Predict opponent behavior
        predicted_highest, predicted_second_highest = self.estimate_opponent_bids(previous_winners, previous_second_highest_bids)
        
        # Calculate the bid
        bid = self.calculate_optimal_bid(current_value, predicted_highest, predicted_second_highest, capital)
        
        # Adjust aggression based on game progress
        self.adjust_aggression(capital, self.round_number)
        
        # Track the bid history
        self.bid_history.append(bid)
        self.capital_history.append(capital)

        return bid
