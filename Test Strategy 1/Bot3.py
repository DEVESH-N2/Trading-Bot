from Strategy import StrategyBase

class UserStrategy(StrategyBase):

    def __init__(self):
        # Track bid history and rounds for more precise predictions
        self.bid_history = []
        self.round_number = 0
        self.aggression_factor = 0.7  # Initial aggression factor (how aggressive we are relative to x_i)
        self.safety_margin = 0.1  # Initial margin for ensuring positive payoff

    def estimate_opponent_bid(self, previous_winners, previous_second_highest_bids):
        '''
        Estimate future bids based on previous winners and second-highest bids.
        The goal is to predict a reasonable bid that will still win but leaves enough margin for a payoff.
        '''
        if not previous_winners or not previous_second_highest_bids:
            return 0, 0  # No data available, can't make a prediction

        avg_highest = sum(previous_winners[-5:]) / len(previous_winners[-5:])
        avg_second_highest = sum(previous_second_highest_bids[-5:]) / len(previous_second_highest_bids[-5:])
        
        return avg_highest, avg_second_highest

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Optimized Strategy for Variation 1:
        - Predict opponent behavior to inform the bid.
        - Use adaptive aggression based on current capital and round number.
        - Ensure bid leaves room for positive payoff.
        '''
        self.round_number += 1
        
        # Step 1: Get predicted opponent behavior
        predicted_highest, predicted_second_highest = self.estimate_opponent_bid(previous_winners, previous_second_highest_bids)
        
        # Step 2: Set a baseline bid
        baseline_bid = current_value * self.aggression_factor  # Bid a fraction of our current value

        # Step 3: Adjust based on predicted opponent behavior
        if predicted_highest > 0:
            # Aim to outbid the predicted second-highest by a small margin
            bid = predicted_second_highest + (predicted_highest - predicted_second_highest) * 0.5
            # Ensure bid isn't higher than our current value minus safety margin
            bid = min(bid, current_value - self.safety_margin)
        else:
            # If no data yet, use a safe conservative bid
            bid = baseline_bid

        # Step 4: Dynamic Aggression & Risk Adjustment
        if capital > 100:  # If capital is abundant, increase aggression
            self.aggression_factor = min(1.0, self.aggression_factor + 0.05)
        elif capital < 30:  # If capital is low, play conservatively
            self.aggression_factor = max(0.5, self.aggression_factor - 0.05)

        # Step 5: Safety Net to Ensure Positive Payoff
        bid = min(bid, current_value - self.safety_margin)

        # Step 6: Capital and Bid Constraints
        bid = min(bid, capital, 100)  # Make sure we don't bid more than we have or the max allowed

        # Track bid history
        self.bid_history.append(bid)

        return bid
