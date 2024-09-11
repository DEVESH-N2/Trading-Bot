import importlib
import os
from Strategy import StrategyBase, StrategyHelper
from random import randint


class Auction:

    def __init__(self, strategy_folder, round_count, starting_capital, max_value, second_highest_fraction, type='self', log=False):
        self.__strategy_folder = strategy_folder  # path where all strategy submissions are located
        self.round_count = round_count  # number of rounds
        self.__round_number = 0
        self.starting_capital = starting_capital
        self.max_value = max_value
        self.type = type
        self.__dead_strategies = 0
        self.log = log
        self.second_highest_fraction = second_highest_fraction
        self.__load_strategies(self.starting_capital, self.max_value)
        self.capitals = [starting_capital for _ in self.__strategies]
        self.final_profits = [0 for _ in self.__strategies]
        self.simulate()

    def __load_strategies(self, starting_capital, max_value):
        self.__strategies = []
        for file in os.listdir(self.__strategy_folder):
            if file.endswith('.py'):
                module_name = file[:-3]  # removes '.py' extension
                module = importlib.import_module(f"{self.__strategy_folder}.{module_name}")
                for attr in dir(module):
                    obj = getattr(module, attr)
                    if isinstance(obj, type) and issubclass(obj, StrategyBase) and obj is not StrategyBase:
                        self.__strategies.append(StrategyHelper(module_name, obj(), starting_capital, max_value, second_highest_fraction=self.second_highest_fraction, log=self.log))

    def __pick_from_distribution(self):
        # Picks a random number from distribution of our choice
        return randint(0, 100)

    def __get_values(self):
        self.values = [self.__pick_from_distribution() for _ in range(len(self.__strategies)) if self.__strategies[_].capital > 0]
        return self.values

    def __get_bids(self):
        return [(strategy.bid(value, len(self.__strategies) - self.__dead_strategies), strategy) for strategy, value in zip([strategy for strategy in self.__strategies if strategy.capital > 0], self.values) if strategy.capital > 0]

    def find_two_highest(self, nums):
        first, second = float('-inf'), float('-inf')
        for num in nums:
            if num > first:
                first, second = num, first
            elif num > second and num != first:
                second = num
        return first, second

    def run_auction(self):
        # Simulates 1 round of the auction
        self.__round_number += 1
        values = self.__get_values()
        bids, strategies = zip(*self.__get_bids())
        winning_bid, second_highest = self.find_two_highest(bids)
        
        if self.log:
            print(f"Round {self.__round_number}:")
            print(f"Values: {values}")
            print(f"Bids: {bids}")
            print(f"Top 2 bids are {winning_bid:0.2f}, {second_highest:0.2f}")
        
        max_value = max(values)
        
        if winning_bid < 0:
            print("SOMETHING WENT WRONG. ALL BOTS EITHER MADE ILLEGAL BIDS OR ARE OUT OF CAPITAL.")
            print("Bids from active bots:", bids)
            print("Values obtained by active bots:", values)
            print("Capitals remaining:", [strategy.capital for strategy in self.__strategies])
            exit(1)

        for strategy in strategies:
            winning_value = max_value if self.type == 'max' else strategy.value
            if self.compare(strategy.bid_value, winning_bid):
                self.__dead_strategies += strategy.update_capital(winning_value, winning_bid, second_highest, winner=True)
            elif self.compare(strategy.bid_value, second_highest):
                self.__dead_strategies += strategy.update_capital(winning_value, winning_bid, second_highest, winner=False, second=True)
            else:
                self.__dead_strategies += strategy.update_capital(winning_value, winning_bid, second_highest)

    def compare(self, value1, value2, epsilon=0.01):
        return abs(value1 - value2) < epsilon

    def simulate(self):
        while self.__round_number < self.round_count:
            self.run_auction()
        self.print_final_profits()

    def print_final_profits(self):
        print("\nFinal Profits:")
        for i, strategy in enumerate(self.__strategies):
            final_profit = strategy.capital - self.starting_capital
            print(f"Bot {i+1} ({strategy.name}): Final Profit = {final_profit:.2f}")


# Example usage
# auction = Auction(strategy_folder='strategy_folder', round_count=1000, starting_capital=100, max_value=100, second_highest_fraction=0.5, type='max', log=True)
