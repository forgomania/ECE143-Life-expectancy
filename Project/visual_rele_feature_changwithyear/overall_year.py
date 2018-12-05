#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 09:56:31 2018

@author: jinqingyuan
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import subplots
import numpy as np
import random
import collections
from scipy.optimize import leastsq
data='/Users/jinqingyuan/Documents/ECE143/projrct/DATA/modified.csv'


def visual_lifechaningbar(data):
    '''
    This function returns a bar plot with global avarage life expectancy changing over 15 yaers.
    We want to get how features changed could influence life expectancy changing.
    '''
    assert isinstance(data,str)
    
    whf = pd.read_csv(data)
    
    '''using feature'''
    gdp = whf.ix[:,17].values
    year = whf.ix[:,2].values
    life = whf.loc[:,['Life expectancy ']].values
    '''Initial'''
    Li = []
    dic_life = {}
    life_expec = []
    Life_with_year = []
    num_year = dict(collections.Counter(year))
    yr = []
    for k,v in num_year.items():
        yr.append(k)
        for i in range(len(year)):
            if k == year[i]:
                Li.append((k,life[i]))    
    '''y:life_expec'''
    for k,v in Li:
        dic_life.setdefault(k, []).append(v)
    Lif = dic_life.items()
    Lif = dict(Lif)
    for k,v in Lif.items():
        sum_life = np.sum(v)
        mean_life = np.mean(v)
        std_life = np.std(v)
        life_expec.append((sum_life,mean_life,std_life))
    for i in range(len(yr)):
        Life_with_year.append((yr[i],life_expec[i]))
    Z =   []
    for i in range(len(Life_with_year)):
        Z.append((yr[i],Life_with_year[i][1][1]))
    
    '''plot'''
    fig = plt.figure()
    X = []
    Y = []
    for i in range(len(Life_with_year)):
        Y.append(Life_with_year[i][1][1])
    for j in range(len(Life_with_year)):
        X.append(Life_with_year[j][0])
    
    plt.plot(X,Y,marker='*',color = 'k',mec='r', mfc='w')
    plt.xlabel('Year',fontsize = 16)
    plt.ylabel('Life expectency', fontsize = 16)
    plt.title('Life expectency Tendency',fontsize = 18)
    plt.bar(X, Y,color = 'green',width = 0.65,label='graph 1',alpha=0.5)
    plt.legend(('Life expectancy','Average/year'))
    plt.savefig('life_year.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return
