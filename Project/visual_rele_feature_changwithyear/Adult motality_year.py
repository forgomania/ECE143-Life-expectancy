#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:47:59 2018

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


def visual_admchaning(data):
    '''
    This function returns a bar plot with global avarage Adult mortality changing over 15 yaers.
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
    
    '''x1:adm''' 
    alt = whf.ix[:,5].values
    year = whf.ix[:,2].values
    yr = []
    A = []
    ADM = []
    ADM_with_year = []
    dic_adm = {}
    num_year = dict(collections.Counter(year))
    for k,v in num_year.items():
        yr.append(k)
        for i in range(len(year)):
            if k == year[i]:
                A.append((k,alt[i]))
              
    for k,v in A:
        dic_adm.setdefault(k, []).append(v)
    AD = dic_adm.items()
    AD = dict(AD)
    for k,v in AD.items():
        sum_gdp = np.sum(v)
        mean_gdp = np.mean(v)
        std_gdp = np.std(v)
        ADM.append((sum_gdp,mean_gdp,std_gdp))
    for i in range(len(yr)):
        ADM_with_year.append((yr[i],ADM[i]))
    X = []
    X_1 = []
    for i in range(len(Life_with_year)):
        g = ADM_with_year[i][1][1]
        X_1.append(g)
    for j in range(len(Life_with_year)):
        X.append(Life_with_year[j][0])
    '''plot'''
    plt.plot(X,X_1,marker='p',color = 'r', mec='g', mfc='w',label=u'World average')
    plt.xlabel('Year',fontsize = 14)
    plt.ylabel('Adult motality',fontsize = 14)
    plt.title('Adult motality Tendency',fontsize = 18)
    plt.bar(X, X_1,color = 'k',width = 0.65,label='graph 1')
    plt.legend(('Adult motality','Average/year'),loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig('adm_year.jpg',dpi=300)
    plt.show()
    plt.close()
    
    return