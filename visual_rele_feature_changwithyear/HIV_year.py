#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:27:48 2018

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


def visual_hivchaning(data):
    '''
    This function returns a bar plot with global avarage BMI changing over 15 yaers.
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
    
    
    '''x1:hiv''' 
    hiv = whf.ix[:,16].values
    year = whf.ix[:,2].values
    yr = []
    H = []
    HIV = []
    HIV_with_year = []
    dic_hiv = {}
    num_year = dict(collections.Counter(year))
    for k,v in num_year.items():
        yr.append(k)
        for i in range(len(year)):
            if k == year[i]:
                H.append((k,hiv[i]))
              
    for k,v in H:
        dic_hiv.setdefault(k, []).append(v)
    HV = dic_hiv.items()
    HV = dict(HV)
    for k,v in HV.items():
        sum_gdp = np.sum(v)
        mean_gdp = np.mean(v)
        std_gdp = np.std(v)
        HIV.append((sum_gdp,mean_gdp,std_gdp))
    for i in range(len(yr)):
        HIV_with_year.append((yr[i],HIV[i]))
    X = []
    X_1 = []
    for i in range(len(Life_with_year)):
        g = HIV_with_year[i][1][1]*1000
        X_1.append(g)
    for j in range(len(Life_with_year)):
        X.append(Life_with_year[j][0])
    '''plot'''
    plt.plot(X,X_1,marker='s',color = 'k', mec='g', mfc='w',label=u'World average')
    plt.xlabel('Year',fontsize = 14)
    plt.ylabel('HIV-ADIS',fontsize = 14)
    plt.title('HIV Tendency',fontsize = 18)
    plt.bar(X, X_1,color = 'r',width = 0.65,label='graph 1',alpha = 0.5)
    plt.legend(('HIV-AIDS','Average/year'),loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig('hiv_year.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return
