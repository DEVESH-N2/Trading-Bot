from Strategy import StrategyBase
import numpy as np

class AdvancedUserStrategy(StrategyBase):

    def __init__(self):
        self.round_number = 0
        self.capital_history = []
        self.bid_history = []
        self.previous_winners = []
        self.previous_second_highest_bids = []
        self.alpha = 0.2
        self.safety_margin = 0.1
        self.model_params = {'mean': 0, 'std': 1}

    def update_statistics(self, previous_winners, previous_second_highest_bids):
        if previous_winners:
            self.model_params['mean'] = np.mean(previous_winners)
            self.model_params['std'] = np.std(previous_winners)

    def estimate_opponent_bids(self, previous_winners, previous_second_highest_bids):
        mean = self.model_params['mean']
        std = self.model_params['std']
        predicted_highest = np.random.normal(mean, std)
        predicted_second_highest = np.random.normal(mean, std)
        return predicted_highest, predicted_second_highest

    def calculate_optimal_bid(self, current_value, predicted_highest, predicted_second_highest, capital):
        bid = predicted_second_highest + (predicted_highest - predicted_second_highest) * 0.5
        bid = min(bid, capital, current_value - self.safety_margin)
        bid = max(bid, 0.1)
        return bid

    def adjust_aggression(self, capital, round_number):
        if capital > 100 and round_number < 500:
            self.alpha = min(0.9, self.alpha + 0.05)
        elif capital < 30 or round_number > 800:
            self.alpha = max(0.1, self.alpha - 0.05)

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        self.round_number += 1
        
        self.update_statistics(previous_winners, previous_second_highest_bids)
        
        predicted_highest, predicted_second_highest = self.estimate_opponent_bids(previous_winners, previous_second_highest_bids)
        
        bid = self.calculate_optimal_bid(current_value, predicted_highest, predicted_second_highest, capital)
        
        self.adjust_aggression(capital, self.round_number)
        
        self.bid_history.append(bid)
        self.capital_history.append(capital)
        self.previous_winners = previous_winners
        self.previous_second_highest_bids = previous_second_highest_bids

        return bid
