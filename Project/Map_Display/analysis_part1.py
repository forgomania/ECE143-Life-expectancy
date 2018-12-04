# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


def showBar(data,feature,year):
    '''
    Used to show a bar plot for the given feature and year, for all countries 
    in the dataframe data.
    
    params:
        data (FataFrame)
        feature (str)
        year (int)
    '''
    assert isinstance(data,pd.core.frame.DataFrame)
    assert isinstance(feature,str)
    assert isinstance(year,int)
    assert data.Year.min() <= year <= data.Year.max()  
    
    data=data[data['Year']==year]
    data=data.sort_index(by=feature)
    bottomValue=list(data.iloc[list(range(10))][feature])
    bottomName=tuple(data.iloc[list(range(10))]['Country'])
    topValue=list(data.iloc[list(range(-10,0))][feature])
    topName=tuple(data.iloc[list(range(-10,0))]['Country'])
    
    plt.figure()
    plt.title(feature+'/ Country')
    y_pos = np.arange(len(bottomName))
    plt.subplot(1, 2, 1)
    plt.bar(y_pos, bottomValue, align='center', alpha=0.5)
    plt.xticks(y_pos, bottomName)
    plt.ylabel(feature)
    plt.ylim(0, 100)
    
    y_pos = np.arange(len(topName))
    plt.subplot(1, 2, 2)
    plt.bar(y_pos, topValue, align='center', alpha=0.5)
    plt.xticks(y_pos, topName)
    plt.ylabel(feature)
    plt.ylim(0, 100)
    plt.show()


def showSpot(data,feature,year,nums):
    """
    Shows the comparison between top countries and bottom countries in life 
    expectancy and the given feature / year.
    
    params:
        data (FataFrame)
        feature (str)
        year (int)    
    """
    assert isinstance(data,pd.core.frame.DataFrame)
    assert isinstance(feature,str)
    assert isinstance(year,int)
    assert data.Year.min() <= year <= data.Year.max()  
    
    data=data[data['Year']==year]
    data=data.sort_index(by='Life expectancy ')
    bottomValue=list(data.iloc[list(range(nums))][feature])
    bottomLE=list(data.iloc[list(range(nums))]['Life expectancy '])
    topValue=list(data.iloc[list(range(-nums,0))][feature])
    topLE=list(data.iloc[list(range(-nums,0))]['Life expectancy '])
    
    plt.figure()
    plt.title(feature+'/ Country')
    plt.scatter(bottomValue,bottomLE,label='bottom'+str(nums))
    plt.xlabel(feature)
    plt.ylabel('Life expectancy ')
    plt.scatter(topValue,topLE,label='top'+str(nums))
    
    plt.legend()
    plt.show()
    #return data
    
def analysis(data):

    assert isinstance(data,pd.core.frame.DataFrame)
        
    data2000=data[data['Year']==2000]
        
    newdata = data2000.sort_index(by='Life expectancy ')
    
    showBar(data,'Life expectancy ',2000)
    showBar(data,'Life expectancy ',2010)
    
    for i in data.columns:
        showSpot(data,i,2000,20)




