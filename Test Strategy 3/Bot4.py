from Strategy import StrategyBase
import numpy as np

class SimplifiedAuctionStrategy(StrategyBase):
    
    def __init__(self):
        self.round_number = 0
        self.history_winners = []
        self.history_second_highest = []
        self.capital_history = []
        self.num_bidders_history = []
        
    def update_statistics(self, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Update historical data.
        '''
        self.history_winners.extend(previous_winners)
        self.history_second_highest.extend(previous_second_highest_bids)
        self.capital_history.append(capital)
        self.num_bidders_history.append(num_bidders)

    def estimate_opponent_bids(self):
        '''
        Use historical data to estimate opponent bids.
        '''
        if self.history_winners:
            mean_winner = np.mean(self.history_winners)
            std_winner = np.std(self.history_winners)
        else:
            mean_winner = 0
            std_winner = 1
        
        # Predict opponents' bids using mean and variance
        predicted_max_bid = mean_winner + 0.5 * std_winner
        predicted_second_highest_bid = mean_winner
        
        return predicted_max_bid, predicted_second_highest_bid

    def calculate_optimal_bid(self, current_value, predicted_max_bid, predicted_second_highest_bid, capital):
        '''
        Determine the optimal bid by balancing risk and potential payoff.
        '''
        # Basic strategy: Bid slightly below the predicted maximum bid
        bid = min(predicted_max_bid - 0.1 * (predicted_max_bid - predicted_second_highest_bid), capital)
        bid = max(bid, 0.1)  # Ensure bid is not too low
        return min(bid, current_value - 0.1)  # Ensure bid is less than current value

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Main method to determine the bid.
        '''
        self.round_number += 1

        # Update statistics with historical data
        self.update_statistics(previous_winners, previous_second_highest_bids, capital, num_bidders)
        
        # Estimate opponent behavior
        predicted_max_bid, predicted_second_highest_bid = self.estimate_opponent_bids()
        
        # Calculate the optimal bid
        bid = self.calculate_optimal_bid(current_value, predicted_max_bid, predicted_second_highest_bid, capital)
        
        return bid
