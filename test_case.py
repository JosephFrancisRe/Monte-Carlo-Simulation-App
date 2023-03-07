# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 03:54:57 2023

@author: JosephRe
"""

from monte_carlo import MonteCarlo as mc
import matplotlib.pyplot as plt

# Run the Monte Carlo application with 10 simulations so the result will be
# easy to understand when plotted.
sim = mc.monte_carlo_sim(10000, .105, .195, 30, 10)

# Create plot with result
plt.plot(sim)
plt.title('Monte Carlo Simulation - Portfolio Value Prediction')
plt.ylabel('Porfolio Value (USD)')
plt.xlabel('Time Horizon (Years)')
plt.savefig('monte_carlo_simulations.png', bbox_inches='tight')