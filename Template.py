from Strategy import StrategyBase

class UserStrategy(StrategyBase):

    # Refer to Strategy.py for implementation details
    # Edit the function below without changing its template(name, input parameters, output type)
    def make_bid(self, current_value, previous_winners,previous_second_highest_bids,capital,num_bidders):
        '''
        Make sure you also keep track of how much capital you have left.
        Any strategy that makes a bid greater than available capital will be immediately discarded
        '''
        # Example strategy: bid 10 more than the previous highest bid.
        if previous_winners:
            return min(100,max(previous_winners) + 10)
        else:
            return 10
        
    # Feel free to make use of class variables and helper functions while editing the above function
        