from Strategy import StrategyBase
import numpy as np

class AdvancedAuctionStrategy(StrategyBase):
    
    def __init__(self):
        self.round_number = 0
        self.alpha = 0.1  # Learning rate for Bayesian updating
        self.safety_margin = 0.05
        self.initial_aggression_factor = 0.5
        self.max_aggression_factor = 0.8
        self.min_aggression_factor = 0.2
        self.aggression_factor = self.initial_aggression_factor
        self.previous_winners = []
        self.previous_second_highest_bids = []
        self.mean_highest_bid = 0
        self.std_highest_bid = 1
        self.mean_second_highest_bid = 0
        self.std_second_highest_bid = 1

    def update_statistics(self, previous_winners, previous_second_highest_bids):
        '''
        Update estimation of opponents' bidding behavior using Bayesian updating.
        '''
        if previous_winners:
            self.mean_highest_bid = np.mean(previous_winners)
            self.std_highest_bid = np.std(previous_winners)
        if previous_second_highest_bids:
            self.mean_second_highest_bid = np.mean(previous_second_highest_bids)
            self.std_second_highest_bid = np.std(previous_second_highest_bids)

    def estimate_opponent_bids(self, num_bidders):
        '''
        Estimate opponent bids using historical data and adjust for the number of bidders.
        '''
        predicted_max_bid = self.mean_highest_bid + self.alpha * self.std_highest_bid
        predicted_second_highest_bid = self.mean_second_highest_bid + self.alpha * self.std_second_highest_bid

        # Adjust predictions based on the number of bidders
        predicted_max_bid = predicted_max_bid * (num_bidders / (num_bidders + 1))
        predicted_second_highest_bid = predicted_second_highest_bid * (num_bidders / (num_bidders + 1))

        return predicted_max_bid, predicted_second_highest_bid

    def calculate_optimal_bid(self, current_value, predicted_max_bid, predicted_second_highest_bid, capital):
        '''
        Calculate the optimal bid by balancing risk and expected payoff.
        '''
        # Adjust aggression factor based on remaining capital and round number
        aggression_range = self.max_aggression_factor - self.min_aggression_factor
        capital_factor = min(1, capital / 100)  # Example factor to adjust aggression based on capital
        round_factor = min(1, self.round_number / 1000)  # Example factor to adjust aggression based on rounds
        self.aggression_factor = self.min_aggression_factor + aggression_range * (capital_factor + round_factor) / 2

        # Base bid calculation
        base_bid = predicted_second_highest_bid + (predicted_max_bid - predicted_second_highest_bid) * self.aggression_factor

        # Adjust bid based on current value and capital
        adjusted_bid = base_bid * (current_value / max(current_value, 1))  # Adjust based on current value
        adjusted_bid = min(adjusted_bid, capital, current_value - self.safety_margin)
        
        return max(adjusted_bid, 0.1)  # Ensure bid is not too low

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Make a bid based on current value and historical data.
        '''
        self.round_number += 1

        # Update statistics with historical data
        self.update_statistics(previous_winners, previous_second_highest_bids)
        
        # Estimate opponent behavior
        predicted_max_bid, predicted_second_highest_bid = self.estimate_opponent_bids(num_bidders)
        
        # Calculate the optimal bid
        bid = self.calculate_optimal_bid(current_value, predicted_max_bid, predicted_second_highest_bid, capital)
        
        # Track historical data for future reference
        self.previous_winners.append(predicted_max_bid)
        self.previous_second_highest_bids.append(predicted_second_highest_bid)
        
        return bid
