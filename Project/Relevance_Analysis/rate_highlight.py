#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 01:29:21 2018
@author: jinqingyuan
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import subplots
import numpy as np
import random
import collections
from scipy.optimize import leastsq

def visual_highlightrate(whf):
    '''
    This part we analyze feature's changing rate over years.
    This fuction will return one particular relevant feature's changing rate which I want
    With the background of other features. 
    '''
       
    whf.insert(0, 'dummy', whf['Measles '])
    '''life'''
    
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
    X = []
    Y = []
    L = []
    for i in range(len(Life_with_year)):
        Y.append(Life_with_year[i][1][1])
    for j in range(len(Life_with_year)):
        X.append(Life_with_year[j][0])
    for i in range(1,len(X)):
        l = ((Y[i-1]-Y[i])/Y[i])*100
        L.append(l)
    L.reverse()
        
    '''x1:gdp''' 
    gdp = whf.ix[:,17].values
    year = whf.ix[:,2].values
    yr = []
    G = []
    GDP = []
    GDP_with_year = []
    dic_gdp = {}
    num_year = dict(collections.Counter(year))
    for k,v in num_year.items():
        yr.append(k)
        for i in range(len(year)):
            if k == year[i]:
                G.append((k,gdp[i]))
              
    for k,v in G:
        dic_gdp.setdefault(k, []).append(v)
    GD = dic_gdp.items()
    GD = dict(GD)
    for k,v in GD.items():
        sum_gdp = np.sum(v)
        mean_gdp = np.mean(v)
        std_gdp = np.std(v)
        GDP.append((sum_gdp,mean_gdp,std_gdp))
    for i in range(len(yr)):
        GDP_with_year.append((yr[i],GDP[i]))
    X = []
    X_1 = []
    L = []
    for i in range(len(Life_with_year)):
        g = GDP_with_year[i][1][1]
        X_1.append(g)
    for j in range(len(Life_with_year)):
        X.append(Life_with_year[j][0])
    
    for i in range(1,len(X)):
        l = ((X_1[i-1]-X_1[i])/X_1[i])*100
        L.append(l)
    L.reverse()
    X = range(2001,2016)
    plt.plot(X,L,marker='',color = 'grey',mec='r', mfc='w',alpha = 0.2)
    plt.xlabel('Year',fontsize = 12)
    plt.ylabel('Changing Ratio(%)', fontsize = 12)
    plt.title('Changing Rate Tendency',fontsize = 18)
    plt.grid(True)
#    plt.plot(X,L,marker='*',color = 'y',mec='y', mfc='w',linewidth=3,alpha = 1)
#    plt.text(2012,8,'GDP',color='y',fontsize=16,withdash=True)
    
    '''x2:income''' 
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
    L=[]
    for i in range(1,len(X)):
        l = ((X_1[i-1]-X_1[i])/X_1[i])*100
        L.append(l)
    L.reverse()
    X = range(2001,2016)
    plt.plot(X,L,marker='',color = 'grey',mec='r', mfc='w',alpha = 0.2)
#    plt.plot(X,L,marker='*',color = 'darkgoldenrod',mec='darkgoldenrod', mfc='w',linewidth=3,alpha = 1)
#    plt.text(2009,4,'Income resources',color='darkgoldenrod',fontsize=16,withdash=True)
    
    
    '''x3:Schooling''' 
    sch = whf.ix[:,22].values
    year = whf.ix[:,2].values
    yr = []
    S = []
    SCH = []
    SCH_with_year = []
    dic_sch = {}
    num_year = dict(collections.Counter(year))
    for k,v in num_year.items():
        yr.append(k)
        for i in range(len(year)):
            if k == year[i]:
                S.append((k,sch[i]))
              
    for k,v in S:
        dic_sch.setdefault(k, []).append(v)
    SC = dic_sch.items()
    SC = dict(SC)
    for k,v in SC.items():
        sum_gdp = np.sum(v)
        mean_gdp = np.mean(v)
        std_gdp = np.std(v)
        SCH.append((sum_gdp,mean_gdp,std_gdp))
    for i in range(len(yr)):
        SCH_with_year.append((yr[i],SCH[i]))
    X = []
    X_1 = []
    for i in range(len(Life_with_year)):
        g = SCH_with_year[i][1][1]
        X_1.append(g)
    for j in range(len(Life_with_year)):
        X.append(Life_with_year[j][0])
    L=[]
    for i in range(1,len(X)):
        l = ((X_1[i-1]-X_1[i])/X_1[i])*100
        L.append(l)
    L.reverse()
    X = range(2001,2016)
    plt.plot(X,L,marker='',color = 'grey',mec='r', mfc='w',alpha = 0.2)
    plt.plot(X,L,marker='*',color = 'dodgerblue',mec='dodgerblue', mfc='w',linewidth=3,alpha = 1)
    plt.text(2012,5,'Schooling',color='dodgerblue',fontsize=16,withdash=True)
    
    '''x4:adm''' 
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
    L=[]
    for i in range(1,len(X)):
        l = ((X_1[i-1]-X_1[i])/X_1[i])*100
        L.append(l)
    L.reverse()
    X = range(2001,2016)
    plt.plot(X,L,marker='',color = 'grey',mec='r', mfc='w',alpha = 0.2)
#    plt.plot(X,L,marker='*',color = 'k',mec='r', mfc='w',linewidth=3,alpha = 1)
#    plt.text(2010,-7,'Adult mortality',color='k',fontsize=16,withdash=True)
    
    
    '''x5:hiv''' 
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
    L=[]
    for i in range(1,len(X)):
        l = ((X_1[i-1]-X_1[i])/X_1[i])*100
        L.append(l)
    L.reverse()
    X = range(2001,2016)
    plt.plot(X,L,marker='',color = 'grey',mec='r', mfc='w',alpha = 0.2)
#    plt.plot(X,L,marker='*',color = 'salmon',mec='salmon', mfc='w',linewidth=3,alpha = 1)
#    plt.text(2013,-8,'HIV',color='salmon',fontsize=16,withdash=True)
    
    '''x6:bmi''' 
    hiv = whf.ix[:,11].values
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
        g = HIV_with_year[i][1][1]
        X_1.append(g)
    for j in range(len(Life_with_year)):
        X.append(Life_with_year[j][0])
    L=[]
    for i in range(1,len(X)):
        l = ((X_1[i-1]-X_1[i])/X_1[i])*100
        L.append(l)
    L.reverse()
    X = range(2001,2016)
    plt.plot(X,L,marker='',color = 'grey',mec='r', mfc='w',alpha = 0.2)
    #plt.plot(X,L,marker='*',color = 'orange',mec='orange', mfc='w',linewidth=3,alpha = 1)
    #plt.text(2012,5,'BMI',color='orange',fontsize=16,withdash=True)
    
    
    '''x7:hb''' 
    hiv = whf.ix[:,9].values
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
    L=[]
    for i in range(1,len(X)):
        l = ((X_1[i-1]-X_1[i])/X_1[i])*100
        L.append(l)
    L.reverse()
    X = range(2001,2016)
    plt.plot(X,L,marker='',color = 'grey',mec='r', mfc='w',alpha = 0.2)
    #plt.plot(X,L,marker='*',color = 'fuchsia',mec='fuchsia', mfc='w',linewidth=3,alpha = 1)
    #plt.text(2011,5,'Hepatitis B',color='fuchsia',fontsize=16,withdash=True)
    
    
    '''save'''
    plt.savefig('rate_schooling.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return