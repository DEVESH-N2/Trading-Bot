from Strategy import StrategyBase
import numpy as np
from scipy.optimize import linprog

class NashEquilibriumStrategy(StrategyBase):

    def __init__(self):
        self.round_number = 0
        self.capital_history = []
        self.bid_history = []
        self.mean_highest_bid = 0
        self.mean_second_highest_bid = 0
        self.num_players = 1  # Initialize with one player (itself)

    def update_opponent_estimates(self, previous_winners, previous_second_highest_bids):
        if previous_winners:
            self.mean_highest_bid = np.mean(previous_winners)
        if previous_second_highest_bids:
            self.mean_second_highest_bid = np.mean(previous_second_highest_bids)
        self.num_players = len(previous_winners) + 1

    def compute_nash_equilibrium(self, current_value, capital):
        num_players = self.num_players
        
        # Create objective function coefficients (all zeros since we only need to satisfy constraints)
        c = np.zeros(num_players)
        
        # Create constraints for bids to be non-negative and within capital
        A_ub = -np.eye(num_players)
        b_ub = -np.ones(num_players) * capital

        # Set bounds for each player's bid to be between 0 and capital
        bounds = [(0, capital) for _ in range(num_players)]

        # Solve linear programming problem to find equilibrium bids
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        # Use the solution if successful, otherwise fallback to conservative bidding
        if res.success:
            equilibrium_bid = min(res.x[0], current_value - 0.1)
        else:
            equilibrium_bid = capital * 0.1
        
        return equilibrium_bid

    def make_bid(self, current_value, previous_winners, previous_second_highest_bids, capital, num_bidders):
        self.round_number += 1

        # Update estimates based on previous rounds
        self.update_opponent_estimates(previous_winners, previous_second_highest_bids)
        
        # Compute the Nash equilibrium bid
        bid = self.compute_nash_equilibrium(current_value, capital)
        
        # Ensure bid is within bounds and not too low
        bid = max(bid, 0.1)
        bid = min(bid, current_value - 0.1)
        
        # Track bidding and capital history
        self.bid_history.append(bid)
        self.capital_history.append(capital)
        
        return bid
