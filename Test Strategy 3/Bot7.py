from Strategy import StrategyBase
import numpy as np
from scipy.stats import norm

class CombinedBidStrategy(StrategyBase):

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

    def calculate_probabilistic_bid(self, mean_winner, std_winner, mean_second_highest, std_second_highest):
        '''
        Calculate a bid based on probabilistic modeling.
        '''
        predicted_max_value = norm.rvs(loc=mean_winner, scale=std_winner)
        predicted_second_highest_bid = norm.rvs(loc=mean_second_highest, scale=std_second_highest)
        
        return predicted_max_value, predicted_second_highest_bid

    def calculate_risk_adjusted_bid(self, current_value, predicted_max_value, predicted_second_highest_bid, capital):
        '''
        Calculate the optimal bid with risk adjustment and aggression management.
        '''
        # Aggression factor with decay
        aggression_factor = max(self.initial_aggression - self.aggression_decay * self.round_number, 0.1)
        
        # Base bid calculation
        base_bid = predicted_second_highest_bid + (predicted_max_value - predicted_second_highest_bid) * aggression_factor
        
        # Adjust bid to be within capital constraints
        adjusted_bid = min(base_bid, capital - self.safety_margin)
        
        return max(adjusted_bid, 0.1)  # Ensure bid is not too low

    def calculate_expected_payoff(self, predicted_max_value, predicted_second_highest_bid, current_value, bid):
        '''
        Calculate the expected payoff based on predicted values and bid.
        '''
        prob_win = 1 - norm.cdf(bid, loc=predicted_max_value, scale=self.max_value / 4)
        expected_payoff = prob_win * (current_value - bid) - (1 - prob_win) * 0.5 * (current_value - bid)
        
        return expected_payoff

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Determine the bid for the current round.
        '''
        self.round_number += 1

        # Update statistics with historical data
        self.update_statistics(previous_winners, previous_second_highest_bids, capital, num_bidders)
        
        # Estimate bid distribution
        mean_winner, std_winner, mean_second_highest, std_second_highest = self.estimate_bid_distribution()
        
        # Calculate probabilistic estimates
        predicted_max_value, predicted_second_highest_bid = self.calculate_probabilistic_bid(mean_winner, std_winner, mean_second_highest, std_second_highest)
        
        # Calculate the optimal bid with risk adjustment
        bid = self.calculate_risk_adjusted_bid(current_value, predicted_max_value, predicted_second_highest_bid, capital)
        
        # Optionally, adjust bid based on expected payoff
        expected_payoff = self.calculate_expected_payoff(predicted_max_value, predicted_second_highest_bid, current_value, bid)
        bid = max(bid, expected_payoff)  # Adjust based on expected payoff
        
        return bid
