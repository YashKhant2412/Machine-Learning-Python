# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:13:29 2020

@author: Yash
"""


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

#Implement UCB
N = 10000
d=10 
ads_selected = []
rewards = []
numbers_of_selection = [0] * d
sums_of_rewards = [0] * d
total_rewards = 0
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if(numbers_of_selection[i]>0):
            avarage_reward = sums_of_rewards[i]/numbers_of_selection[i]
            delta_i = math.sqrt((3*math.log(n+1))/(2*numbers_of_selection[i]))
            upper_bound = avarage_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound>max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selection[ad] +=1 
    reward = dataset.values[n,ad]
    sums_of_rewards[ad]+=reward
    total_rewards+=reward
    rewards.append(reward)

#Visulization
plt.hist(ads_selected)
plt.title("Histogram of ads Selection")
plt.xlabel('ads')
plt.ylabel('Numbers of time selected')
plt.savefig('ads.svg')
plt.show()





        

