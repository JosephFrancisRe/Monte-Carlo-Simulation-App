# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 03:54:57 2023

@author: JosephRe
"""

from monte_carlo import MonteCarlo as mc
import matplotlib.pyplot as plt

sim = mc.monte_carlo_sim(10000, .105, .195, 30, 10)
plt.plot(sim)
plt.title('Monte Carlo Simulation - Stock Price Prediction')
plt.ylabel('Stock Price (USD)')
plt.xlabel('Time Horizon (Years)')
plt.savefig('monte_carlo_simulations.png', bbox_inches='tight')