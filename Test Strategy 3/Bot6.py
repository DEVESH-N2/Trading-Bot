from Strategy import StrategyBase
import numpy as np

class AdvancedBidStrategy(StrategyBase):

    def __init__(self):
        self.round_number = 0
        self.history_winners = []
        self.history_second_highest = []
        self.capital_history = []
        self.num_bidders_history = []
        self.safety_margin = 0.1
        self.initial_aggression = 0.5
        self.aggression_decay = 0.01
        self.alpha = 0.1  # Smoothing factor for estimation
        self.max_value = 100  # Maximum bid value

    def update_statistics(self, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Update historical data with new round information.
        '''
        self.history_winners.extend(previous_winners)
        self.history_second_highest.extend(previous_second_highest_bids)
        self.capital_history.append(capital)
        self.num_bidders_history.append(num_bidders)

    def estimate_bid_distribution(self):
        '''
        Estimate the distribution of opponent bids using historical data.
        '''
        if len(self.history_winners) > 0:
            mean_winner = np.mean(self.history_winners)
            std_winner = np.std(self.history_winners)
            mean_second_highest = np.mean(self.history_second_highest)
            std_second_highest = np.std(self.history_second_highest)
        else:
            mean_winner = self.max_value / 2
            std_winner = self.max_value / 4
            mean_second_highest = self.max_value / 2
            std_second_highest = self.max_value / 4
        
        return mean_winner, std_winner, mean_second_highest, std_second_highest

    def calculate_risk_adjusted_bid(self, current_value, mean_winner, std_winner, mean_second_highest, std_second_highest, capital):
        '''
        Calculate the optimal bid with probabilistic adjustment and risk management.
        '''
        # Estimate potential maximum value (X)
        predicted_max_value = mean_winner + std_winner * np.random.randn()

        # Estimate potential second-highest bid
        predicted_second_highest_bid = mean_second_highest + std_second_highest * np.random.randn()

        # Aggression factor with decay
        aggression_factor = max(self.initial_aggression - self.aggression_decay * self.round_number, 0.1)
        
        # Calculate base bid
        base_bid = predicted_second_highest_bid + (predicted_max_value - predicted_second_highest_bid) * aggression_factor
        
        # Adjust bid to be within capital constraints
        adjusted_bid = min(base_bid, capital - self.safety_margin)
        
        return max(adjusted_bid, 0.1)  # Ensure bid is not too low

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Determine the bid for the current round.
        '''
        self.round_number += 1

        # Update statistics with historical data
        self.update_statistics(previous_winners, previous_second_highest_bids, capital, num_bidders)
        
        # Estimate bid distribution
        mean_winner, std_winner, mean_second_highest, std_second_highest = self.estimate_bid_distribution()
        
        # Calculate the optimal bid with risk adjustment
        bid = self.calculate_risk_adjusted_bid(current_value, mean_winner, std_winner, mean_second_highest, std_second_highest, capital)
        
        return bid
