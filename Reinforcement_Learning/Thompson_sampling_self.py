# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 09:11:56 2020

@author: Yash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('Ads_CTR_Optimisation.csv')


#Thomson Sampling
import random
N=10000 
d = 10
ads_selected = []
number_of_rewards_1 = [0]*d
number_of_rewards_0 = [0]*d
total_rewards = 0
for n in range(0,N):
    ad = 0
    max_random = 0
    for i in range(0,d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1,number_of_rewards_0[i]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = data.values[n,ad]
    if reward==1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
    total_rewards = total_rewards+reward


#Visulize
plt.hist(ads_selected)
plt.title("Histogram of ads Selection")
plt.xlabel('ads')
plt.ylabel('Numbers of time selected')
plt.savefig('ads.svg')
plt.show()






