#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 00:34:29 2018
@author: jinqingyuan
"""


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import subplots
import numpy as np
import random
import collections
from scipy.optimize import leastsq

def visual_allfeaturechaning(whf):
    '''
    This function returns a bar plot with global avarage GDP changing over 15 yaers.
    We want to get how features changed could influence life expectancy changing.
    '''
    #data='/output/modified.csv'
    whf.insert(0, 'dummy', whf['Measles '])
    
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
    for i in range(len(Life_with_year)):
        g = GDP_with_year[i][1][1]
        X_1.append(g)
    for j in range(len(Life_with_year)):
        X.append(Life_with_year[j][0])
    plt.plot(X,X_1,marker='^', color = 'k',mec='g', mfc='w',label=u'World average')
    plt.xlabel('Year',fontsize = 14)
    plt.ylabel('GDP/500',fontsize = 14)
    plt.title('GDP Tendency',fontsize = 18)
    plt.bar(X, X_1,color ='y',width = 0.65)
    plt.legend(('GDP','Average/year'),loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig('GDP_year.jpg',dpi=500)
    plt.show()
    plt.close()
        
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
    ax = plt.plot(X,X_1,marker='h',color = 'k', mec='g', mfc='w',label=u'World average')
    plt.xlabel('Year from 2000 to 2015',fontsize = 16)
    plt.ylabel('Income resources',fontsize = 16)
    plt.bar(X, X_1,color = 'darkgoldenrod',width = 0.65)
    plt.legend(('Income resources','Average/year'),loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig('Income_year.jpg',dpi=500)
    plt.show()
    plt.close()

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
    ax = plt.plot(X,X_1,marker='o',color = 'k', mec='g', mfc='w',label=u'World average')
    plt.xlabel('Year',fontsize = 14)
    plt.ylabel('Schooling',fontsize = 14)
    plt.title('Schooling Tendency',fontsize = 18)
    plt.bar(X, X_1,width = 0.65,label='')
    plt.legend(('Schooling','Average/year'),loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig('school_year.jpg',dpi=500)
    plt.show()
    plt.close()    

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
    plt.plot(X,X_1,marker='p',color = 'r', mec='g', mfc='w',label=u'World average')
    plt.xlabel('Year',fontsize = 14)
    plt.ylabel('Adult motality',fontsize = 14)
    plt.title('Adult motality Tendency',fontsize = 18)
    plt.bar(X, X_1,color = 'k',width = 0.65,label='graph 1')
    plt.legend(('Adult motality','Average/year'),loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig('adm_year.jpg',dpi=300)
    plt.show()
    plt.close()

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
    plt.plot(X,X_1,marker='s',color = 'k', mec='g', mfc='w',label=u'World average')
    plt.xlabel('Year',fontsize = 14)
    plt.ylabel('HIV-ADIS',fontsize = 14)
    plt.title('HIV Tendency',fontsize = 18)
    plt.bar(X, X_1,color = 'r',width = 0.65,label='graph 1',alpha = 0.5)
    plt.legend(('HIV-AIDS','Average/year'),loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig('hiv_year.jpg',dpi=500)
    plt.show()
    plt.close()

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
    '''plot'''
    plt.plot(X,X_1,marker='d',color = 'k', mec='g', mfc='w',label=u'World average')
    plt.xlabel('Year',fontsize = 14)
    plt.ylabel('HIV-ADIS',fontsize = 14)
    plt.title('BMI Tendency',fontsize = 18)
    plt.bar(X, X_1,color = 'orange',width = 0.65,label='graph 1')
    plt.legend(('BMI','Average/year'),loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig('BMI_year.jpg',dpi=500)
    plt.show()
    plt.close()
    
    '''x7:hiv''' 
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
    plt.plot(X,X_1,marker='d',color = 'k', mec='g', mfc='w',label=u'World average')
    plt.yticks(np.arange(0, 100000, 20000))
    plt.xlabel('Year',fontsize = 14)
    plt.ylabel('HIV-ADIS',fontsize = 14)
    plt.title('Hepatitis B Tendency',fontsize = 18)
    plt.bar(X, X_1,color = 'magenta',width = 0.65,label='graph 1',alpha=0.5)
    plt.legend(('Hepatitis B','Average/year'),loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig('Hb_year.jpg',dpi=500)
    plt.show()
    plt.close()        
        
    return