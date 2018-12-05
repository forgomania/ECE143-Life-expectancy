#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:53:32 2018

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


def visual_lifechaning(data):
    '''
    This function returns a curve plot with global avarage life expectancy changing over 15 yaers.
    We want to get how features changed could influence life expectancy changing.
    '''
    assert isinstance(data,str)
    
    whf = pd.read_csv(data)
    
    '''using feature'''
    
    gdp = whf.ix[:,17].values
    total_expenditure = whf.ix[:,14].values
    status = whf.ix[:,3].values
    country = whf.ix[:,1].values
    year = whf.ix[:,2].values
    life = whf.loc[:,['Life expectancy ']].values
    '''Initial'''
    G = []
    Ex = []
    S = []
    Li = []
    dic_gdp = {}
    dic_expen = {}
    dic_status = {}
    dic_life = {}
    GDP = []
    EXPENDI = []
    STATUS = []
    life_expec = []
    GDP_with_country = []
    EXPENDI_with_country = []
    STATUS_with_country = []
    Life_with_year = []
    num_year = dict(collections.Counter(year))
    yr = []
    for k,v in num_year.items():
        yr.append(k)
        for i in range(len(year)):
            if k == year[i]:
                G.append((k,gdp[i]))
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
            
    '''plot'''
    X = []
    Y = []
    Z = []
    for i in range(len(Life_with_year)):
        Z.append((yr[i],Life_with_year[i][1][1]))
    for i in range(len(Life_with_year)):
        Y.append(Life_with_year[i][1][1])
    for j in range(len(Life_with_year)):
        X.append(Life_with_year[j][0])
    
    
    plt.plot(X,Y,marker='*', mec='g', mfc='w',label=u'World average')
    plt.xlabel('Year from 2000 to 2015',fontsize = 16)
    plt.ylabel('Life expectency',fontsize = 16)
    plt.legend(('each year',''),loc='upper left')