# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 03:50:10 2023

@author: JosephRe
"""

import numpy as np
import numpy.random as npr
import pandas as pd
from scipy.stats import norm

class MonteCarlo:
    '''
    A class that can perform a variety of calculations related to Monte Carlo.

    Methods
    -------
    quick_var(value, vol, T, CL)
        Calculates the deterministic parametric value at risk

    mc_VaR(pv, er, vol, T, iterations)
        Calculates a Monte Carlo estimation of parametric value at risk

    monte_carlo_sim(pv, er, volatility, horizon, iterations)
        Performs Monte Carlo simulations to calculate expected portfolio ending value
    '''
    
    def deterministic_var(value, vol, T, confidence):
        '''
        Parametric value at risk calculation (also known as the deterministic approach).
        Volatility is proportial to the square root of time and a percent point function
        is used as the probability distribution.
        
        Parameters:
        value (float): Value of portfolio in USD
        vol (float): Portfolio volatility measure
        T (int): Number of years
        confidence (float): Confidence level in the percent point function distribution
        
        Returns:
        numpy.float64: Value at risk in USD for one simple 'simulation'
        '''
        cutoff = norm.ppf(confidence)
        return value * vol * np.sqrt(T) * cutoff


    def mc_VaR(pv, er, vol, T, iterations):
        '''
        Parametric value at risk calculation (also known as the variance-covariance method).
        Uses Monte Carlo probabilistic methods that leverage the standard normal probability
        distribution to compute the portfolio's maximum loss. Based on Black-Scholes method.
        
        Parameters:
        pv (float): Value of portfolio in USD
        er (float): Expected return for the portfolio
        vol (float): Portfolio volatility measure
        T (int): Number of years
        iterations (int): Number of iterations
        
        Returns:
        numpy.ndarray: Vector containing the value at risk estimation
        '''
        end = pv * np.exp((er - .5 * vol ** 2) * T + 
                     vol * np.sqrt(T) * np.random.standard_normal(iterations))
        ending_values = end - pv
        return ending_values


    def monte_carlo_sim(pv, er, vol, horizon, iterations):
        '''
        Calculates portfolio ending value using Monte Carlo simulations and returns the
        result in a format that can be viewed or plotted.
        
        Parameters:
        pv (float): Value of portfolio in USD
        er (float): Expected return for the portfolio
        vol (float): Portfolio volatility measure
        horizon (int): Time horizon in years
        iterations (int): Number of iterations
        
        Returns:
        pandas.DataFrame: DataFrame containing the all simulations results for portfolio ending value
        '''  
        returns = np.zeros((iterations, horizon))
        for t in range(iterations):
            for year in range(horizon):
                returns[t][year] = npr.normal(er, vol)
        portfolio = np.zeros((iterations,horizon))
        for iteration in range(iterations):
            starting = pv
            for year in range(horizon):
                ending = starting * (1 + returns[iteration,year])
                portfolio[iteration,year] = ending
                starting = ending
        return pd.DataFrame(portfolio).T