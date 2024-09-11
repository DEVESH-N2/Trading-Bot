from Strategy import StrategyBase

class UserStrategy(StrategyBase):

    def __init__(self):
        # Tracking the bid history and capital for adaptive strategy
        self.bid_history = []
        self.round_number = 0
        self.aggression_factor = 0.8  # How aggressively we want to bid (0.8 means 80% of current value)

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        '''
        Enhanced Strategy for Variation 1:
        - Adaptive bidding based on previous winners and current value.
        - Balances aggression based on capital and round progression.
        - Ensures bid leaves room for positive payoff.
        '''
        
        self.round_number += 1
        
        # Step 1: Set a baseline bid based on current value
        baseline_bid = current_value * self.aggression_factor  # Start by bidding 80% of current value

        # Step 2: Adapt based on previous winners
        if previous_winners:
            # Average of the last 5 winning bids to determine the trend in bidding
            avg_previous_highest = sum(previous_winners[-5:]) / len(previous_winners[-5:])
            
            # Calculate an adaptive bid based on previous winners (slightly higher than average)
            adaptive_bid = avg_previous_highest + (current_value - avg_previous_highest) * 0.5  # Halfway between avg and current value
            
            # Step 3: Adjust aggression dynamically based on current capital and number of rounds
            if capital < 30:  # If capital is low, bid conservatively
                bid = min(baseline_bid, adaptive_bid)
            elif self.round_number > 500:  # In late-game, bid more aggressively to maximize payoffs
                bid = adaptive_bid
            else:
                # Normal situation, bid conservatively but still competitive
                bid = max(baseline_bid, adaptive_bid)
        else:
            # Early rounds: Stick to a safe, conservative bid
            bid = baseline_bid

        # Step 4: Ensure bid is within capital and max bid constraints
        bid = min(bid, capital, 100)

        # Track the bid for future analysis
        self.bid_history.append(bid)

        return bid
