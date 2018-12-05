#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:35:58 2018

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


def visual_incomechaning(data):
    '''
    This function returns a bar plot with global avarage income composition changing over 15 yaers.
    We want to get how features changed could influence life expectancy changing.
    '''
    assert isinstance(data,str)
    
    whf = pd.read_csv(data)

    
    '''using feature'''
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
    
    
    
    '''x1:income''' 
    inc = whf.ix[:,21].values
    year = whf.ix[:,2].values
    yr = []
    I = []
    INC = []
    INC_with_year = []
    dic_inc = {}
    num_year = dict(collections.Counter(year))
    for k,v in num_year.items():
        yr.append(k)
        for i in range(len(year)):
            if k == year[i]:
                I.append((k,inc[i]))
              
    for k,v in I:
        dic_inc.setdefault(k, []).append(v)
    IC = dic_inc.items()
    IC = dict(IC)
    for k,v in IC.items():
        sum_gdp = np.sum(v)
        mean_gdp = np.mean(v)
        std_gdp = np.std(v)
        INC.append((sum_gdp,mean_gdp,std_gdp))
    for i in range(len(yr)):
        INC_with_year.append((yr[i],INC[i]))
    X = []
    X_1 = []
    for i in range(len(Life_with_year)):
        g = INC_with_year[i][1][1]
        X_1.append(g)
    for j in range(len(Life_with_year)):
        X.append(Life_with_year[j][0])
    '''plot'''
    ax = plt.plot(X,X_1,marker='h',color = 'k', mec='g', mfc='w',label=u'World average')
    plt.xlabel('Year from 2000 to 2015',fontsize = 16)
    plt.ylabel('Income resources',fontsize = 16)
    plt.bar(X, X_1,color = 'darkgoldenrod',width = 0.65)
    plt.legend(('Income resources','Average/year'),loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig('Income_year.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return