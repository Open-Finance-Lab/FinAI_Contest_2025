# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:24:34 2024
@author: ab978
"""
import pdb
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import talib
from talib import MA_Type

'''
PROBLEM STATEMENT:  a liquidity provider (LP) holds capital and want to participate in
an AMM Uniswap v3 pool. She can adjust the position at discrete hourly time steps.

It is an optimal uniform allocation strategy around the current price of the token pair.

There has to be a mapping: action is discrete integer i, then current price is p. 
The interval will be [current_tick - i, current_tick + i] which then translates to
[p_l,p_u]

state variables:
    - tech idx
    - USD in portfolio
    - width of liquidity interval (previous action)
    - value of liquidity position at t in USD
    - central tick of the price interval
    
Value of liquidity position can be initialized in several ways (varying initial funds)

To map ticks to prices one needs to implement the formula at page 161 of Ottina book

Action space is discrete from 0 to N where N is the max width of the liquidity range allowed.

Reward is the result of liquidity reallocation

'''

class Uniswapv3Env(gym.Env):
    """
    A custom environment for simulating interaction in a Uniswapv3 AMM.
    
    Attributes:
        delta (float): The fee tier of the AMM.
        n_actions (int): Choices for price range width
        market_data (np.ndarray): The preorganized data from a pandas DataFrame, used for simulation.
        d (float): The tick spacing of the AMM.
        x (int): Initial quantity of asset X (ETH)
        gas (float): fixed gas fee
    """
    
    def __init__(self, 
                 delta: float, 
                 action_values: np.array,
                 market_data: pd.DataFrame,
                 x: int,
                 gas: float):
        super(Uniswapv3Env, self).__init__()
        # store array of preorganized data from a pandas dataframe
        prices = market_data.values.astype('float64')
        self.market_data = prices[:,0]
        # store the column names of the pandas dataframe
        self.names = market_data.columns.tolist()
        self.names.extend(['m','w','l','sigma','ma24','ma168','bb_upper','bb_middle','bb_lower'])
        self.names.extend(['ADXR', 'BOP', 'DX'])
        self.delta = delta
        
        self.d = self._fee_to_tickspacing(self.delta) # tick spacing
        
        # action space
        self.action_space = spaces.Discrete(len(action_values))
        self.action_values = action_values
        
        self.w = self.action_values[1] # initial interval width
        
        # gas fee
        self.gas = gas
        self.x = x
        self.initial_x = x
        
        # Boundaries to choose
        lower_bounds = []
        upper_bounds = []
        for name in self.names:
            lower_bounds.append(-np.inf)
            upper_bounds.append(np.inf)
        lower_bounds = np.array(lower_bounds, dtype=np.float32)
        upper_bounds = np.array(upper_bounds, dtype=np.float32)
        # Define the observation space as a continuous vector space
        self.observation_space = spaces.Box(low=lower_bounds, high=upper_bounds, shape=(len(self.names),), dtype=np.float32)
        # To test it run "self.observation_space.contains(np.array([1,2,1,1,1,1],dtype=np.float32))"
        
        # Initialize current state
        self.current_state = None
        
        # calculate EW sigma
        log_returns = np.log(self.market_data[1:] / self.market_data[:-1])
        df = pd.DataFrame({'return': log_returns.flatten()})
        ew_sigma = df.ewm(alpha=0.05).std()
        self.ew_sigma = ew_sigma['return'].to_numpy()
        
        # calculate the 12-hr high, low, open, and close price for p_t
        # using p_t and prior 11 prices
        self.open_price, self.high_price, self.low_price, self.closed_price = self._calculate_candles(12)
        
        ma_window_0 = 24
        ma_window_max = 168
        # calculate moving average MA24 and MA168
        self.moving_average_24 = market_data.rolling(ma_window_0).mean()
        self.moving_average_168 = market_data.rolling(ma_window_max).mean()
        
        # cut the first 168 entries (NaN for MA data)
        self.moving_average_24 = self.moving_average_24.to_numpy()
        self.moving_average_24 = self.moving_average_24[ma_window_max:]
        self.moving_average_168 = self.moving_average_168.to_numpy()
        self.moving_average_168 = self.moving_average_168[ma_window_max:]
        
        # calculate bollinger bands, with triple exponential moving average:
        bb_upper, bb_middle, bb_lower = talib.BBANDS(self.market_data, matype=MA_Type.T3)
        self.bb_upper = bb_upper[ma_window_max:]
        self.bb_middle = bb_middle[ma_window_max:]
        self.bb_lower = bb_lower[ma_window_max:]
        # print("BB_UPPER:", len(self.market_data), len(self.bb_upper))
        
        # self.open_price = self.open_price[ma_window_max:]
        # self.high_price = self.high_price[ma_window_max:]
        # self.low_price = self.low_price[ma_window_max:]
        # self.closed_price = self.closed_price[ma_window_max:]
        
        # calculate ADXR, BOP, DX
        self.adxr = talib.ADX(self.high_price, self.low_price, self.closed_price, timeperiod=14)    # Average Directional Movement Index Rating
        self.adxr = self.adxr[ma_window_max:]
        self.bop = talib.BOP(self.open_price, self.high_price, self.low_price, self.closed_price)   # Balance Of Power
        self.bop = self.bop[ma_window_max:]
        self.dx = talib.DX(self.high_price, self.low_price, self.closed_price, timeperiod=14)       # Directional Movement Index
        self.dx = self.dx[ma_window_max:]
        
        self.market_data = self.market_data[ma_window_max:]
        self.ew_sigma = self.ew_sigma[ma_window_max-1:] # We use a -1 because we compute the ew_sigma on the log returns, which are shifted by one time step.
            
    def reset(self, **kwargs):
    
        self.history = []
        
        
        self.count = 0 # iteration counter
        self.cumul_reward = 0
        self.cumul_fee = 0
        
        # Assuming self.market_data[0] is a NumPy array
        pt = self.market_data[self.count]
        m = self._price2tick(pt)  # Convert price to tick
        
        # initialize liquidity
        tl, tu = m - self.d*self.w, m + self.d*self.w
        pl, pu = self._tick2price(tl), self._tick2price(tu)
        self.pl = pl
        self.pu = pu
        # print("RESET")
        # print("p_u: ", pu, " p_l: ", pl)
        # print("Initial x: ", self.x)
        self.x = self.initial_x
        self.l = self.x / (1/np.sqrt(pt) - 1/np.sqrt(pu))
        # print("Initial liquidity: ", self.l)
        
        # initialize y
        self.y = self.l * (np.sqrt(pt) - np.sqrt(pl))
        # print("Initial y: ", self.y)
        
        ma24 = self.moving_average_24[self.count, 0]
        ma168 = self.moving_average_168[self.count, 0]
        
        # record data
        self.history.append({
            'X': self.x,
            'Y': self.y,
            'Liquidity': self.l,
            'Price': pt,
            'Price_Upper': pu,
            'Price_Lower': pl,
            'Gas': self.gas,
            'Action': self.w,
            'Sigma': 1,
            'Width': self.w,
            'Reward': 0,
            'Fee': 0,
            'LVR': 0,
            'ma24': ma24,
            'ma168': ma168,
            'Value': self.x * pt + self.y,
        })
        
        states = [self.market_data[self.count], m, self.w, self.l, 1]
        states.extend([ma24, ma168, self.bb_upper[self.count], self.bb_middle[self.count], self.bb_lower[self.count]])
        states.extend([self.adxr[self.count], self.bop[self.count], self.dx[self.count]])
        self.current_state = np.array(states)

        # # Ensure the returned observation is within the observation_space
        # if not self.observation_space.contains(self.current_state):
        #     raise ValueError("The observation returned by the step() method is not within the observation space.")

        
        return self.current_state, {} # a dict of info is needed and I initialized it empty
    
    def step(self, action_index):
        """
        Advances the environment by one step based on the given action.

        Parameters:
            action (int): The action taken by the external RL agent, representing an interest rate adjustment.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: A tuple containing the next observation, the calculated reward, done flag, and additional info.
        """
        action = self.action_values[action_index]
        
        # 0
        # update count
        self.count += 1
        
        # record xt_1 and yt_1
        xt_1 = self.x
        yt_1 = self.y
        
        # 1
        # calculate tick corresponding to AMM market price
        # pt_1: price 1h before
        # pt:   price now
        pt_1, pt = self.market_data[self.count-1], self.market_data[self.count]
        
        # 2
        # calculate xt and yt
        # based on pt and old price interval
        # without changing liquidity
        m = self._price2tick(pt)
        pl_1, pu_1 = self.pl, self.pu
        xt, yt = self._calculate_xy(pt, pl_1, pu_1)
        
        # 2.1 
        # if xt or yt is already 0 after 1 hour of market evolution
        # we have to reposition the LP
        if xt == 0 or yt == 0:
            # print("pt out of the (pl_1, pu_1)")
            # if action is 0, force action to be 1
            # reset the LP set interval width +/- 1
            if action == 0:
                pl = self.pl
                pu = self.pu
                # calculate liquidity
                if xt == 0 and yt != 0:
                    self.l = yt / (np.sqrt(pu) - np.sqrt(pl))
                elif yt == 0 and xt != 0:
                    self.l = xt / (1/np.sqrt(pl) - 1/np.sqrt(pu))
                else:   # xt = 0 and yt = 0
                    self.l = 0
                
            else:
                # calculate new interval based on action
                self.w = action
                tl, tu = m - self.d*self.w, m + self.d*self.w
                pl, pu = self._tick2price(tl), self._tick2price(tu)  
                
                # calculate new X and Y based pt
                if yt == 0:
                    # print("X: ", xt, " Y:  0", "Reset Y")
                    xt = xt/2
                    yt = xt * pt
                elif xt == 0:
                    # print("X:  0  Y:  ", yt, "Reset X")
                    yt = yt/2
                    xt = yt/pt
                    
                # update liquidity
                self.l = xt / (1/np.sqrt(pt) - 1/np.sqrt(pu))
            
        # 2.2 
        # price not out of old interval
        else:
            if action != 0:
                self.w = action
                # calculate new pl, pu
                tl, tu = m - self.d*self.w, m + self.d*self.w
                pl, pu = self._tick2price(tl), self._tick2price(tu)   

                # (pull all xt, yt out of the pool)
                # inject the same amount of xt and yt back to the pool
                # calculate new liquidity
                self.l = xt / (1/np.sqrt(pt) - 1/np.sqrt(pu))
                
            else:
                # action = 0, we do not need to do anything
                # update pl, pu
                pl = pl_1
                pu = pu_1

        # update self.x and self.y to xt and yt
        self.x = xt
        self.y = yt
        
        # reward as per the original paper
        gas_fee = self._indicator(action)*self.gas
        
        # calculate fees
        if pt_1 <= pt:
            if pt_1 >= pu or pt <= pl:
                fees = 0
            else:
                p_prime = np.minimum(pt, pu)
                p = np.maximum(pt_1, pl)
                fees = self._calculate_fee(p, p_prime)
        else:
            if pt_1 <= pl or pt >= pu:
                fees = 0
            else:
                p_prime = np.maximum(pt, pl)
                p = np.minimum(pt, pu)
                fees = self._calculate_fee(p, p_prime)
        
        sigma = self.ew_sigma[self.count]   
        
        vp = self.x * pt + self.y
        ll = self.l * sigma * sigma / 4 * np.sqrt(pt)
        if vp != 0:
            lvr = ll
        else:
            lvr = 1e+9
            
        # if self.x == 0 or self.y == 0:
        #     lvr = 0
        
        # print("fee: ", fees, ", LVR: ", lvr)
        # print("Gas Fee: ", gas_fee, " Fee: ", fees, " LVR: ", lvr)
        reward = - gas_fee + fees - lvr
        self.cumul_reward += reward
        self.cumul_fee += fees
        
        # consider ma24 and ma24*7=168, ma 24*30 and ma 24*60
        # pre-train the moving average and cut the data
        # plot and share the plot on teams
        # retry the hyperparameters without seed varying
        
        # try exponentially weighted moving average
        ma24 = self.moving_average_24[self.count, 0]
        ma168 = self.moving_average_168[self.count, 0]
        
        self.initial_value = self.x * pt + self.y

        # record data
        self.history.append({
            'X': self.x,
            'Y': self.y,
            'Liquidity': self.l,
            'Price': pt,
            'Price_Upper': pu,
            'Price_Lower': pl,
            'Gas': gas_fee,
            'Action': action,
            'Sigma': sigma,
            'Width': self.w,
            'Reward': reward,
            'Fee': fees,
            'LVR': lvr,
            'ma24': ma24,
            'ma168': ma168,
            'Value': self.initial_value
        })
        
        self.pl = pl
        self.pu = pu
        
        bb_count = self.count
        if self.count >= len(self.bb_upper):
            bb_count = len(self.bb_upper) - 1
        
        states = [self.market_data[self.count], m, self.w, self.l, sigma]
        states.extend([ma24, ma168, self.bb_upper[bb_count], self.bb_middle[bb_count], self.bb_lower[bb_count]])
        states.extend([self.adxr[self.count], self.bop[self.count], self.dx[self.count]])
        self.current_state = np.array(states)
        
        terminated = self.count >= self.market_data.shape[0] - 2
        truncated = self.l <= 1e-9
            
        # if truncated:
        #     print(self.count, "Truncated here", self.l, "Liquidity")
        
        # truncated = False
        info = {}

        # # Ensure the returned observation is within the observation_space
        # if not self.observation_space.contains(self.current_state):
        #     raise ValueError("The observation returned by the step() method is not within the observation space.")

        return self.current_state, reward, terminated, truncated, info
    
    def _calculate_candles(self, interval_length: int):
        open_prices = []
        high_prices = []
        low_prices = []
        close_prices = []

        interval_length = interval_length - 1

        # Iterate through market data
        for i in range(len(self.market_data)):
            if i < interval_length:
                # For the first 0-10 (11) data points, calculate the i-hour candle
                start_idx = 0
            else:
                # For the rest, use the last 12 hours of data
                start_idx = i - interval_length

            # Get the relevant data slice
            data_slice = self.market_data[start_idx : i+1]
            
            # Calculate open, high, low, and close
            open_price = data_slice[0]
            high_price = max(data_slice)
            low_price = min(data_slice)
            close_price = data_slice[-1]

            # Append the results to their respective arrays
            open_prices.append(open_price)
            high_prices.append(high_price)
            low_prices.append(low_price)
            close_prices.append(close_price)
            
        # Convert lists to NumPy arrays with dtype float64
        open_prices = np.array(open_prices, dtype=np.float64)
        high_prices = np.array(high_prices, dtype=np.float64)
        low_prices = np.array(low_prices, dtype=np.float64)
        close_prices = np.array(close_prices, dtype=np.float64)

        return open_prices, high_prices, low_prices, close_prices

    def _price2tick(self, p: float):
        return math.floor(math.log(p, 1.0001))
    
    def _tick2price(self, t: int):
        return 1.0001**t
    
    def _fee_to_tickspacing(self, fee_tier: float):
        if fee_tier == 0.05:
            return 10
        elif fee_tier == 0.30:
            return 60
        elif fee_tier == 1.00:
            return 200
        else:
            raise ValueError(f"Unsupported fee tier: {fee_tier}")
            
    def _calculate_fee(self, p, p_prime):
        if p <= p_prime:
            # fee = (self.delta / (1 - self.delta)) * self.l * (math.sqrt(p) - math.sqrt(p_prime))
            fee = (self.delta / (1 - self.delta)) * self.l * (math.sqrt(p_prime) - math.sqrt(p))
        else:
            # fee = (self.delta / (1 - self.delta)) * self.l * ((1 / math.sqrt(p)) - (1 / math.sqrt(p_prime))) * p_prime
            fee = (self.delta / (1 - self.delta)) * self.l * ((1 / math.sqrt(p_prime)) - (1 / math.sqrt(p))) * p_prime
        return fee
    
    def _indicator(self,a):
        return 1 if a != 0 else 0
    
    def _calculate_xy(self, p, pl, pu):
        # Ottina et al. Page 169
        if p <= pl:
            x = self.l * (1 / math.sqrt(pl) - 1 / math.sqrt(pu))
            y = 0
        elif p >= pu:
            x = 0
            y = self.l * (math.sqrt(pu) - math.sqrt(pl))
        else:  # pl < p < pu
            x = self.l * (1 / math.sqrt(p) - 1 / math.sqrt(pu))
            y = self.l * (math.sqrt(p) - math.sqrt(pl))
        return x, y
    
    def _calculate_ma(self, array, count, window_size):
        # https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/
        if window_size > count:
            window_size = count
            
        window = array[count - window_size : count]
        window_average = np.round(np.sum(window) / window_size, 2)
        
        return window_average


class CustomMLPFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom MLP feature extractor.
    """
    def __init__(self, observation_space: spaces.Box, 
                 features_dim: int = 256,
                 hidden_dim: list = [8,8],
                 activation: str = 'relu'):
        super(CustomMLPFeatureExtractor, self).__init__(observation_space, features_dim)

        if activation == 'relu':
            act = nn.ReLU()
        else:
            act = nn.Tanh()
        
        self.net = nn.Sequential(
            nn.BatchNorm1d(observation_space.shape[0], affine=False),
            nn.Linear(observation_space.shape[0], hidden_dim[0]),
            act,
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            act,
            nn.Linear(hidden_dim[1], features_dim)  # Connects to both policy and value networks in Stable Baselines3
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
