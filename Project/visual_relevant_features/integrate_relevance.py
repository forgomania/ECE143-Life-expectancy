#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 00:49:15 2018

@author: jinqingyuan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import collections
from scipy.optimize import leastsq

'''Population'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/14-Population.csv'
def visualpopulation(data):
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    
    popf = pd.read_csv(data)
    population = popf.ix[:,4].values
    country = popf.ix[:,1].values
    life = popf.loc[:,['Life expectancy ']].values
    '''Initial'''
    P = []
    Li = []
    dict_population = {}
    dic_life = {}
    POPULA = []
    life_expec = []
    POPULA_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                P.append((k,population[i]))
                Li.append((k,life[i]))
                
    '''x:population'''           
    for k,v in P:
        dict_population.setdefault(k, []).append(v)
    POP = dict_population.items()
    POP = dict(POP)
    for k,v in POP.items():
        sum_population = np.sum(v)
        mean_population = np.mean(v)
        std_population = np.std(v)
        POPULA.append((sum_population,mean_population,std_population))
    for i in range(len(cntry)):
        POPULA_with_country.append((cntry[i],POPULA[i]))

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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(POPULA_with_country)):
        X.append(POPULA_with_country[j][1][1])
    
    np.random.seed(19680801)
    areaX =[]
    for i in X:
        j = i/5000000
        areaX.append(j)
    
    plt.scatter(X,Y,s=areaX, c='g', alpha=0.5)
    plt.xlabel('Population',fontsize = 16)
    plt.ylabel('Life expectency',fontsize = 16)
    plt.legend(('each country',''),loc='upper right')
    plt.savefig('popula_rele.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return


'''Measles'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/6-Measles .csv'
def visualmeasles(data):
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)   
    
    '''using feature'''
    schf = pd.read_csv(data)    
    sch = schf.ix[:,4].values
    country = schf.ix[:,1].values
    life = schf.loc[:,['Life expectancy ']].values
    '''Initial'''
    Sch = []
    Li = []
    dic_sch = {}
    dic_life = {}
    SCH = []
    life_expec = []
    SCH_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                Sch.append((k,sch[i]))
                Li.append((k,life[i]))
    '''x: measles'''           
    for k,v in Sch:
        dic_sch.setdefault(k, []).append(v)
    CH = dic_sch.items()
    CH = dict(CH)
    for k,v in CH.items():
        sum_SCH = np.sum(v)
        mean_SCH = np.mean(v)
        std_SCH = np.std(v)
        SCH.append((sum_SCH,mean_SCH,std_SCH))
    for i in range(len(cntry)):
        SCH_with_country.append((cntry[i],SCH[i]))
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(SCH_with_country)):
        X.append(SCH_with_country[j][1][1])
    areaX =[]
    for i in X:
        j = i/150
        areaX.append(j)
    plt.scatter(X,Y,s=areaX, c='g', alpha=0.5)
    plt.xlabel('Measles',fontsize = 12)
    plt.ylabel('Life expectency',fontsize = 12)
    plt.title('Relevance with Measles',fontsize = 18)
    plt.legend(('each country',''),loc='upper right')
    plt.savefig('mesal_rele.jpg',dpi=500)
    plt.show()
    plt.close()
    return

'''Infant death'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/2-infant deaths.csv'
def visualinfantdeath(data):
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    
    '''using feature'''
    infdf = pd.read_csv(data)
    infd = infdf.ix[:,4].values
    country = infdf.ix[:,1].values
    life = infdf.loc[:,['Life expectancy ']].values
    '''Initial'''
    IN = []
    Li = []
    dic_sch = {}
    dic_life = {}
    INF = []
    life_expec = []
    INF_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                IN.append((k,infd[i]))
                Li.append((k,life[i]))
                
    '''x: infantdeath'''           
    for k,v in IN:
        dic_sch.setdefault(k, []).append(v)
    CH = dic_sch.items()
    CH = dict(CH)
    for k,v in CH.items():
        sum_SCH = np.sum(v)
        mean_SCH = np.mean(v)
        std_SCH = np.std(v)
        INF.append((sum_SCH,mean_SCH,std_SCH))
    for i in range(len(cntry)):
        INF_with_country.append((cntry[i],INF[i]))
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(INF_with_country)):
        X.append(INF_with_country[j][1][1])
    
    np.random.seed(19680801)
    areaX =[]
    for i in X:
        j = i/3
        areaX.append(j)
    
    plt.scatter(X,Y,s=areaX, c='g', alpha=0.5)
    plt.xlabel('Infant death',fontsize = 12)
    plt.ylabel('Life expectency',fontsize = 12)
    plt.title('Relevance with Hepatitis B',fontsize = 18)
    plt.legend(('each country',''),loc='upper right')
    plt.savefig('infant death_rele.jpg',dpi=500)
    plt.show()
    plt.close()
    return


'''Alcohol'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/3-Alcohol.csv'
def visualAlcohol(data):
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    schf = pd.read_csv(data)
    sch = schf.ix[:,4].values
    country = schf.ix[:,1].values
    life = schf.loc[:,['Life expectancy ']].values
    '''Initial'''
    Sch = []
    Li = []
    dic_sch = {}
    dic_life = {}
    SCH = []
    life_expec = []
    SCH_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                Sch.append((k,sch[i]))
                Li.append((k,life[i]))
                
    '''x: schooling'''           
    for k,v in Sch:
        dic_sch.setdefault(k, []).append(v)
    CH = dic_sch.items()
    CH = dict(CH)
    for k,v in CH.items():
        sum_SCH = np.sum(v)
        mean_SCH = np.mean(v)
        std_SCH = np.std(v)
        SCH.append((sum_SCH,mean_SCH,std_SCH))
    for i in range(len(cntry)):
        SCH_with_country.append((cntry[i],SCH[i]))
        
        
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(SCH_with_country)):
        X.append(SCH_with_country[j][1][1])
    
    np.random.seed(19680801)
    areaX =[]
    for i in X:
        j = (i)**2
        areaX.append(j)
    
    plt.scatter(X,Y,s=areaX, c='g', alpha=0.5)
    plt.xlabel('Alcohol',fontsize = 14)
    plt.ylabel('Life expectency',fontsize = 14)
    plt.title('Relevance with alcohol',fontsize = 18)
    plt.savefig('alcohol_rele.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return


'''hepatits B'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/5-Hepatitis B.csv'
def visualhepatitsb(data):
    
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    
    hebf = pd.read_csv(data)
    heb = hebf.ix[:,4].values
    country = hebf.ix[:,1].values
    life = hebf.loc[:,['Life expectancy ']].values
    '''Initial'''
    He = []
    Li = []
    dic_heb = {}
    dic_life = {}
    HEB = []
    life_expec = []
    HEB_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                He.append((k,heb[i]))
                Li.append((k,life[i]))
                
    '''x:HEB'''           
    for k,v in He:
        dic_heb.setdefault(k, []).append(v)
    HB = dic_heb.items()
    HB = dict(HB)
    for k,v in HB.items():
        sum_HEB = np.sum(v)
        mean_HEB = np.mean(v)
        std_HEB = np.std(v)
        HEB.append((sum_HEB,mean_HEB,std_HEB))
    for i in range(len(cntry)):
        HEB_with_country.append((cntry[i],HEB[i]))
        
        
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(HEB_with_country)):
        X.append(HEB_with_country[j][1][1])
    
    np.random.seed(19680801)
    areaX =[]
    for i in X:
        j = (i/10)**2
        areaX.append(j)
    
    plt.xlabel('Hepatitis B',fontsize = 14)
    plt.ylabel('Life expectency',fontsize = 14)
    plt.title('Relevance with Hepatitis B',fontsize = 18)
    nbins = 20
    plt.hist2d(X,Y,bins=nbins,cmap=plt.cm.BuGn_r)
    cbar = plt.colorbar()
    cbar.set_label('Number of countries')
    plt.savefig('Hb_rele.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return


'''BMI'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/7- BMI .csv'
def visualbmi(data):
    
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    
    bmif = pd.read_csv(data)
    bmi = bmif.ix[:,4].values
    country = bmif.ix[:,1].values
    life = bmif.loc[:,['Life expectancy ']].values
    '''Initial'''
    B = []
    Li = []
    dic_bmi = {}
    dic_life = {}
    BMI = []
    life_expec = []
    BMI_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                B.append((k,bmi[i]))
                Li.append((k,life[i]))
                
    '''x:BMI'''           
    for k,v in B:
        dic_bmi.setdefault(k, []).append(v)
    BM = dic_bmi.items()
    BM = dict(BM)
    for k,v in BM.items():
        sum_BMI = np.sum(v)
        mean_BMI = np.mean(v)
        std_BMI = np.std(v)
        BMI.append((sum_BMI,mean_BMI,std_BMI))
    for i in range(len(cntry)):
        BMI_with_country.append((cntry[i],BMI[i]))
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(BMI_with_country)):
        X.append(BMI_with_country[j][1][1])
    
    np.random.seed(19680801)
    areaX =[]
    for i in X:
        j = i**3/2000
        areaX.append(j)
    plt.xlabel('BMI',fontsize = 16)
    plt.ylabel('Life expectency',fontsize = 16)
    plt.title('Relevance with BMI',fontsize = 18)
    nbins = 20
    plt.hist2d(X,Y,bins=nbins,cmap=plt.cm.BuGn_r)
    cbar = plt.colorbar()
    cbar.set_label('Number of countries')
    plt.savefig('bmi_rele.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return


'''Diphtheria'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/11-Diphtheria .csv'
def visualDiph(data):
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    
    schf = pd.read_csv(data)
    sch = schf.ix[:,4].values
    country = schf.ix[:,1].values
    life = schf.loc[:,['Life expectancy ']].values
    '''Initial'''
    Sch = []
    Li = []
    dic_sch = {}
    dic_life = {}
    SCH = []
    life_expec = []
    SCH_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                Sch.append((k,sch[i]))
                Li.append((k,life[i]))
                
    '''x: Diphtheria'''           
    for k,v in Sch:
        dic_sch.setdefault(k, []).append(v)
    CH = dic_sch.items()
    CH = dict(CH)
    for k,v in CH.items():
        sum_SCH = np.sum(v)
        mean_SCH = np.mean(v)
        std_SCH = np.std(v)
        SCH.append((sum_SCH,mean_SCH,std_SCH))
    for i in range(len(cntry)):
        SCH_with_country.append((cntry[i],SCH[i]))
        
        
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(SCH_with_country)):
        X.append(SCH_with_country[j][1][1])
    
    np.random.seed(19680801)
    areaX =[]
    for i in X:
        j = (i/10)**2
        areaX.append(j)
    
    plt.xlabel('Diphtheria',fontsize = 14)
    plt.ylabel('Life expectency',fontsize = 14)
    plt.title('Relevance with Diphtheria',fontsize = 18)
    nbins = 20
    plt.hist2d(X,Y,bins=nbins,cmap=plt.cm.BuGn_r)
    cbar = plt.colorbar()
    cbar.set_label('Number of countries')
    plt.savefig('Dip_rele.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return

'''Thinnese'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/15- thinness  1-19 years.csv'
def visualthinnese(data):
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    
    schf = pd.read_csv(data)
    sch = schf.ix[:,4].values
    country = schf.ix[:,1].values
    life = schf.loc[:,['Life expectancy ']].values
    '''Initial'''
    Sch = []
    Li = []
    dic_sch = {}
    dic_life = {}
    SCH = []
    life_expec = []
    SCH_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                Sch.append((k,sch[i]))
                Li.append((k,life[i]))
                
    '''x: thinness  1-19'''           
    for k,v in Sch:
        dic_sch.setdefault(k, []).append(v)
    CH = dic_sch.items()
    CH = dict(CH)
    for k,v in CH.items():
        sum_SCH = np.sum(v)
        mean_SCH = np.mean(v)
        std_SCH = np.std(v)
        SCH.append((sum_SCH,mean_SCH,std_SCH))
    for i in range(len(cntry)):
        SCH_with_country.append((cntry[i],SCH[i]))
        
        
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(SCH_with_country)):
        X.append(SCH_with_country[j][1][1])
    
    np.random.seed(19680801)
    areaX =[]
    for i in X:
        j = i**2/2
        areaX.append(j)
    
    plt.xlabel('thinness  1-19',fontsize = 14)
    plt.ylabel('Life expectency',fontsize = 14)
    plt.title('Relevance with 1-19 thinnese',fontsize = 18)
    nbins = 20
    plt.hist2d(X,Y,bins=nbins,cmap=plt.cm.BuGn_r)
    cbar = plt.colorbar()
    cbar.set_label('Number of countries')
    plt.savefig('thinnese_rele.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return

'''Total expenditure'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/10-Total expenditure.csv'
def visualATotalexpenditure(data):
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    
    expenf = pd.read_csv(data)
    expenditure = expenf.ix[:,4].values
    country = expenf.ix[:,1].values
    life = expenf.loc[:,['Life expectancy ']].values
    '''Initial'''
    Ex = []
    Li = []
    dic_expen = {}
    dic_life = {}
    EXPENDI = []
    life_expec = []
    EXPENDI_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                Ex.append((k,expenditure[i]))
                Li.append((k,life[i]))               
    '''x:total expenditure'''           
    for k,v in Ex:
        dic_expen.setdefault(k, []).append(v)
    EXP = dic_expen.items()
    EXP = dict(EXP)
    for k,v in EXP.items():
        sum_expenditure = np.sum(v)
        mean_expenditure = np.mean(v)
        std_expenditure = np.std(v)
        EXPENDI.append((sum_expenditure,mean_expenditure,std_expenditure))
    for i in range(len(cntry)):
        EXPENDI_with_country.append((cntry[i],EXPENDI[i]))
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(EXPENDI_with_country)):
        X.append(EXPENDI_with_country[j][1][1])
    areaX =[]
    for i in X:
        j = i*5
        areaX.append(j)
    plt.xlabel('Total expenditure',fontsize = 14)
    plt.ylabel('Life expectency',fontsize = 14)
    plt.title('Relevance with Total expenditure',fontsize = 18)
    nbins = 20
    plt.hist2d(X,Y,bins=nbins,cmap=plt.cm.BuGn_r)
    cbar = plt.colorbar()
    cbar.set_label('Number of countries')
    plt.savefig('total exp_rele.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return


'''Adult Mortality'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/1-Adult Mortality.csv'
def visualAdultMortality(data):
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    
    schf = pd.read_csv(data)
    sch = schf.ix[:,4].values
    country = schf.ix[:,1].values
    life = schf.loc[:,['Life expectancy ']].values
    '''Initial'''
    A = []
    ADM_with_year = []
    Li = []
    dic_sch = {}
    dic_life = {}
    ADM = []
    life_expec = []
    ADM_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                A.append((k,sch[i]))
                Li.append((k,life[i]))   
    '''x: AdultMortality'''           
    for k,v in A:
        dic_sch.setdefault(k, []).append(v)
    CH = dic_sch.items()
    CH = dict(CH)
    for k,v in CH.items():
        sum_SCH = np.sum(v)
        mean_SCH = np.mean(v)
        std_SCH = np.std(v)
        ADM.append((sum_SCH,mean_SCH,std_SCH))
    for i in range(len(cntry)):
        ADM_with_country.append((cntry[i],ADM[i]))
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(ADM_with_country)):
        X.append(ADM_with_country[j][1][1])
    
    areaX =[]
    for i in X:
        j = (i/20)**2
        areaX.append(j)
    
    plt.xlabel('Adult Motality',fontsize = 16)
    plt.ylabel('Life expectency',fontsize = 16)
    plt.title('Relevance with adult motality',fontsize = 18)
    plt.hist2d(X,Y,bins=20,cmap=plt.cm.BuGn_r)
    cbar = plt.colorbar()
    cbar.set_label('Number of countries')
    plt.savefig('Adm_rele.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return


'''Schooling'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/18-Schooling.csv'
def visualSchooling(data):
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    
    schf = pd.read_csv(data)
    sch = schf.ix[:,4].values
    country = schf.ix[:,1].values
    life = schf.loc[:,['Life expectancy ']].values
    '''Initial'''
    Sch = []
    Li = []
    dic_sch = {}
    dic_life = {}
    SCH = []
    life_expec = []
    SCH_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                Sch.append((k,sch[i]))
                Li.append((k,life[i]))
                
    '''x: schooling'''           
    for k,v in Sch:
        dic_sch.setdefault(k, []).append(v)
    CH = dic_sch.items()
    CH = dict(CH)
    for k,v in CH.items():
        sum_SCH = np.sum(v)
        mean_SCH = np.mean(v)
        std_SCH = np.std(v)
        SCH.append((sum_SCH,mean_SCH,std_SCH))
    for i in range(len(cntry)):
        SCH_with_country.append((cntry[i],SCH[i]))
        
        
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(SCH_with_country)):
        X.append(SCH_with_country[j][1][1])
    
    areaX =[]
    for i in X:
        j = i**2/3
        areaX.append(j)
    
    plt.xlabel('Scholing',fontsize = 14)
    plt.title('Relevance with schooling',fontsize = 18)
    plt.hist2d(X,Y,bins=20,cmap=plt.cm.BuGn_r)
    cbar = plt.colorbar()
    cbar.set_label('Number of countries')
    plt.savefig('Shooling_rele.jpg',dpi=500)
    plt.show()
    plt.close()    
    
    return


'''Income'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/17-Income composition of resources.csv'
def visualincome(data):
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)

    incf = pd.read_csv(data)
    
    inc = incf.ix[:,4].values
    country = incf.ix[:,1].values
    life = incf.loc[:,['Life expectancy ']].values
    '''Initial'''
    I = []
    Li = []
    dic_inc = {}
    dic_life = {}
    INC = []
    life_expec = []
    INC_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                I.append((k,inc[i]))
                Li.append((k,life[i]))
                
    '''x: schooling'''           
    for k,v in I:
        dic_inc.setdefault(k, []).append(v)
    CH = dic_inc.items()
    CH = dict(CH)
    for k,v in CH.items():
        sum_SCH = np.sum(v)
        mean_SCH = np.mean(v)
        std_SCH = np.std(v)
        INC.append((sum_SCH,mean_SCH,std_SCH))
    for i in range(len(cntry)):
        INC_with_country.append((cntry[i],INC[i]))
        
        
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(INC_with_country)):
        X.append(INC_with_country[j][1][1])
    
    areaX =[]
    for i in X:
        j = (i*15)**2
        areaX.append(j)
    
    plt.xlabel('Income resources',fontsize = 16)
    plt.ylabel('Life expectency',fontsize = 16)
    plt.title('Relevance with Income resources',fontsize = 18)
    nbins = 20
    plt.hist2d(X,Y,bins=nbins,cmap=plt.cm.BuGn_r)
    cbar = plt.colorbar()
    cbar.set_label('Number of countries')
    plt.savefig('Income_rele.jpg',dpi=500)
    plt.show()
    plt.close()  
    
    return


'''HIV'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/feature_tables/12- HIV-AIDS.csv'
def visualhiv(data):
    '''
    Input:
    data(datatype:'str'):seperated feature datasheet
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    
    hivf = pd.read_csv(data)
    hiv = hivf.ix[:,4].values
    country = hivf.ix[:,1].values
    life = hivf.loc[:,['Life expectancy ']].values
    '''Initial'''
    H = []
    Li = []
    dic_hiv = {}
    dic_life = {}
    HIV = []
    life_expec = []
    HIV_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                H.append((k,hiv[i]))
                Li.append((k,life[i]))
                
    '''x:HIV'''           
    for k,v in H:
        dic_hiv.setdefault(k, []).append(v)
    HV = dic_hiv.items()
    HV = dict(HV)
    for k,v in HV.items():
        sum_HIV = np.sum(v)
        mean_HIV = np.mean(v)
        std_HIV = np.std(v)
        HIV.append((sum_HIV,mean_HIV,std_HIV))
    for i in range(len(cntry)):
        HIV_with_country.append((cntry[i],HIV[i]))
        
        
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X = []
    Y = []
    for i in range(len(Life_with_country)):
        Y.append(Life_with_country[i][1][1])
    for j in range(len(HIV_with_country)):
        X.append(HIV_with_country[j][1][1])
    
    np.random.seed(19680801)
    areaX =[]
    for i in X:
        j = i**1.5
        areaX.append(j)
    
    plt.scatter(X,Y,s=areaX, c='g', alpha=0.5)
    plt.xlabel('HIV',fontsize = 14)
    plt.ylabel('Life expectency',fontsize = 14)
    plt.legend(('each country',''),loc='upper right')
    plt.title('Relevance with HIV',fontsize = 18)
    plt.savefig('HIV_rele.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return


'''GDP'''
data = '/Users/jinqingyuan/Documents/ECE143/projrct/DATA/modified.csv'
def visualGDP(data):
    '''
    Input:
    data(datatype:'str'):dataset is used is modified one
    Output:
    graph(type:'jpg'):how is this feature relevant with life expectancy
    '''
    assert isinstance(data,str)
    
    whf = pd.read_csv(data)

    gdp = whf.ix[:,17].values
    population = whf.ix[:,18].values
    status = whf.ix[:,3].values
    country = whf.ix[:,1].values
    life = whf.loc[:,['Life expectancy ']].values
    
    '''Initial'''
    G = []
    POP = [] 
    S = []
    Li = []
    dic_gdp = {}
    dic_expen = {}
    dic_status = {}
    dic_life = {}
    GDP = []
    POPULA = []
    STATUS = []
    life_expec = []
    GDP_with_country = []
    POPULA_with_country = []
    STATUS_with_country = []
    Life_with_country = []
    num_country = dict(collections.Counter(country))
    cntry = []
    for k,v in num_country.items():
        cntry.append(k)
        for i in range(len(country)):
            if k == country[i]:
                G.append((k,gdp[i]))
                Li.append((k,life[i]))
                POP.append((k,population[i]))
                S.append((k,status[i]))
                
    '''x1:gdp'''           
    for k,v in G:
        dic_gdp.setdefault(k, []).append(v)
    GD = dic_gdp.items()
    GD = dict(GD)
    for k,v in GD.items():
        sum_gdp = np.sum(v)
        mean_gdp = np.mean(v)
        std_gdp = np.std(v)
        GDP.append((sum_gdp,mean_gdp,std_gdp))
    for i in range(len(cntry)):
        GDP_with_country.append((cntry[i],GDP[i]))
        
    
    '''x2:expenditure'''           
    for k,v in POP:
        dic_expen.setdefault(k, []).append(v)
    EXP = dic_expen.items()
    EXP = dict(EXP)
    for k,v in EXP.items():
        sum_expen = np.sum(v)
        mean_expen = np.mean(v)
        std_expen = np.std(v)
        POPULA.append((sum_expen,mean_expen,std_expen))
    for i in range(len(cntry)):
        POPULA_with_country.append((cntry[i],POPULA[i]))
        
        
    '''x3:Status'''         
    for k,v in S:
        dic_status.setdefault(k, []).append(v)
    SAT = dic_status.items()
    SAT = dict(SAT)
    for k,v in SAT.items():
        STATUS.append(v)
    for i in range(len(cntry)):
        STATUS_with_country.append((cntry[i],STATUS[i][0]))
    
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
    
    for i in range(len(cntry)):
        Life_with_country.append((cntry[i],life_expec[i]))
            
    '''plot'''
    X_1 = []
    Xc_1 =[]
    X_2 = []
    Xc_2= []
    X_3 = []
    Xc_3= []
    Y =   []
    Yc=   []
    Z =   []
    Zc = []
    GDP_cap = []
    GDPc_cap = []
    for i in range(len(Life_with_country)):
        if STATUS_with_country[i][1] == 1:
            X_1.append(GDP_with_country[i][1][1])
            X_2.append(POPULA_with_country[i][1][1])
            X_3.append(cntry[i])
            Y.append(Life_with_country[i][1][1])
            Z.append((cntry[i],GDP_with_country[i][1][1],POPULA_with_country[i][1][1],Life_with_country[i][1][1]))
            GDP_cap.append(GDP_with_country[i][1][1])
    for i in range(len(Life_with_country)):
        if STATUS_with_country[i][1] == 2:
            Xc_1.append(GDP_with_country[i][1][1])
            Xc_2.append(POPULA_with_country[i][1][1])
            Xc_3.append(cntry[i])
            Yc.append(Life_with_country[i][1][1])
            Zc.append((cntry[i],GDP_with_country[i][1][1],POPULA_with_country[i][1][1],Life_with_country[i][1][1]))
            GDPc_cap.append(GDP_with_country[i][1][1])
            
    '''scale GDP on points size'''
    areaX1 =[]
    for i in GDP_cap:
        j = i/150
        areaX1.append(j)
    areaX2 =[]
    for i in GDPc_cap:
        j = i/150
        areaX2.append(j)
    plt.scatter(GDP_cap,Y,s=areaX1, c='r', alpha=0.5)
    plt.hold(True)
    plt.scatter(GDPc_cap,Yc,s=areaX2, c='g', alpha=0.5)
    plt.xlabel('GDP/captia',fontsize = 14)
    plt.ylabel('Life expectency',fontsize = 14)
    plt.legend(('developing','developed'),markerscale=0.5,loc='lower right')
    plt.title('Life Expectancy with GDP/captia',fontsize = 16)
    plt.text(79000,83,'Luxem',color='darkgoldenrod',withdash=True)
    plt.text(70000,78,'Norway',color='darkgoldenrod',withdash=True)
    plt.text(54000,82,'Switz',color='darkgoldenrod',withdash=True)
    plt.text(57000,73,'Qatar',color='darkgoldenrod',withdash=True)
    plt.text(50000,76,'USA',color='darkgoldenrod',withdash=True)
    plt.text(2901,80,'CHN',color='k',withdash=True)
    plt.savefig('GDPcaptia.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return

