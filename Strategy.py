from abc import ABC, abstractmethod
import time
import psutil
import multiprocessing

class StrategyBase(ABC):

    @abstractmethod
    def make_bid(self, current_value, previous_winners,previous_second_highest_bids,capital,num_bidders):
        """
        Method to decide the bid amount.
        :param current_round: The current value of the product being auctioned.
        :param previous_bids: List of winning bids from previous rounds.
        :return: The bid amount for the current round.
        """
        pass

class StrategyHelper:
    
    def __init__(self,name,strategy,starting_capital,max_value,second_highest_fraction=0.3,log = True,info_size=100):
        self.log = log
        self.name = name
        self.strategy = strategy
        self.capital = starting_capital
        self.max_value = max_value
        self.previous_winners = []
        self.previous_second_highest = []
        self.status=0
        self.second_highest_fraction = second_highest_fraction
        self.info_size = info_size

        
    def is_valid_bid(self,bid):
        if not (isinstance(bid,float) or isinstance(bid,int)) or bid is None:
            return -1 # invalid return
        elif bid < 0 or bid > self.max_value:
            return -2 # out of range
        elif bid > self.capital:
            return self.capital #  insufficient capital
        else:
            return bid

    def bid(self,current_value,num_bidders):
        self.value = current_value

        bid = self.strategy.make_bid(self.value,self.previous_winners,self.previous_second_highest,self.capital,num_bidders)
        # bid = self.strategy.make_bid(current_value,self.previous_winners,self.previous_second_highest,self.capital,num_bidders)
        self.bid_value = self.is_valid_bid(bid)
        return self.bid_value
    

    # What all i need. Am i highest, am i second highest, max of all values, winning bid, second winning bid
    def update_capital(self,winning_value,highest_bid,second_highest_bid,winner=False,second=False):
        capital_at_start_of_round = self.capital
        if winner:
            self.capital += winning_value - highest_bid
        elif second:
            self.capital -= min(self.capital,self.second_highest_fraction*max(0,winning_value-highest_bid))
        self.previous_winners.append(highest_bid)
        self.previous_second_highest.append(second_highest_bid)
        if len(self.previous_winners) > self.info_size:
            self.previous_winners.pop(0)
            self.previous_second_highest.pop(0)

        if self.log:
            print(f"{self.name}: bid - {self.bid_value:.2f}, initial_capital - {capital_at_start_of_round:.2f}, value - {self.value:.2f}, capital left - {self.capital:.2f} ")
        
        if capital_at_start_of_round>0 and self.capital<=0: #Bot just died
            if self.log: print(f"{self.name} just ran out of capital!")
            self.bid_value = float('-inf')
            self.capital=0
            return 1 # indicates that this bidder can no longer participate in the auction
        else: return 0 