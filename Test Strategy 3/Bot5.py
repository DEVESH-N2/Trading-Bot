from Strategy import StrategyBase
import numpy as np

class ProbabilisticBidStrategy(StrategyBase):

    def __init__(self):
        self.round_number = 0
        self.history_winners = []
        self.history_second_highest = []
        self.capital_history = []
        self.num_bidders_history = []
        self.safety_margin = 0.1
        self.initial_aggression = 0.5
        self.aggression_decay = 0.01
        self.alpha = 0.1  # Smoothing factor for probabilistic estimation

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
            mean_winner = 0
            std_winner = 1
            mean_second_highest = 0
            std_second_highest = 1
        
        return mean_winner, std_winner, mean_second_highest, std_second_highest

    def calculate_risk_adjusted_bid(self, current_value, mean_winner, std_winner, mean_second_highest, std_second_highest, capital):
        '''
        Calculate the optimal bid with probabilistic adjustment and risk management.
        '''
        # Probabilistic adjustment
        prob_threshold = 0.5
        prob_win = 1 - prob_threshold  # Adjust as needed
        
        # Aggression factor with decay
        aggression_factor = self.initial_aggression * (1 - self.round_number / 1000) + self.aggression_decay * (capital / 100)
        
        # Estimate the range of potential winning bids
        predicted_max_bid = mean_winner + std_winner * np.random.randn()
        predicted_second_highest_bid = mean_second_highest + std_second_highest * np.random.randn()

        # Base bid calculation
        base_bid = predicted_second_highest_bid + (predicted_max_bid - predicted_second_highest_bid) * aggression_factor

        # Apply risk management: Limit bid to avoid overbidding
        adjusted_bid = base_bid * (current_value / max(current_value, 1))  # Normalize based on current value
        adjusted_bid = min(adjusted_bid, capital - self.safety_margin)  # Ensure bid is within capital constraints
        
        return max(adjusted_bid, 0.1)  # Ensure bid is not too low

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Determine the bid for the current round.
        '''
        self.round_number += 1

        # Update statistics with historical data
        self.update_statistics(previous_winners, previous_second_highest_bids, capital, num_bidders)
        
        # Estimate opponent bids distribution
        mean_winner, std_winner, mean_second_highest, std_second_highest = self.estimate_bid_distribution()
        
        # Calculate the optimal bid with risk adjustment
        bid = self.calculate_risk_adjusted_bid(current_value, mean_winner, std_winner, mean_second_highest, std_second_highest, capital)
        
        return bid
