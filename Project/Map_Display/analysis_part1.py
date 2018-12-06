# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
#pip install selenium

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

def sortdata(data,year,nums):
    """
    Return data in specific year with only given feature and life expectancy of top and bottom [num] countries.
    Use altair in the main jupyter notebook to display the plots
    params:
        data(DataFrame)
        year(int)
        nums(int)
    """
    assert isinstance(data,pd.core.frame.DataFrame)
    assert isinstance(year,int)
    assert data.Year.min() <= year <= data.Year.max()  
    
    data=data[data['Year']==year]
    data=data.sort_index(by='Life expectancy ')
    
    topdata=data[-nums:].copy()
    topdata['rank']='Top'+str(nums)
    bottomdata=data[0:nums].copy()
    bottomdata['rank']='Bottom'+str(nums)
    return pd.concat((topdata,bottomdata))
    

def showSpot(data,feature,year,nums):
    """
    Shows the comparison between top countries and bottom countries in life 
    expectancy and the given feature / year.
    
    params:
        data (DataFrame)
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
    
    #testdata=data[0:10]
    #testchart=Chart(data)
    #testchart.mark_point().encode(x=feature,y='Life expectancy ',color='Country')
    #testchar=Chart(data)
    #testchar.mark_point().encode(x='GDP',y='Life expectancy ',color='Country')
    #testchar.save('chart.png')
    #return data
    
def analysis(data):
    '''
    Does the part 1 of our data analysis, using the functions above
    
    params:
        data (DataFrame)
    '''

    assert isinstance(data,pd.core.frame.DataFrame)
    
    data2000=data[data['Year']==2000]
        
    newdata = data2000.sort_index(by='Life expectancy ')
    
    showBar(data,'Life expectancy ',2000)
    showBar(data,'Life expectancy ',2010)
    
    for i in data.columns:
        showSpot(data,i,2000,20)



def showScatter(data,x,size=None,y='Life expectancy ',year=2000, nums=20):
    
    rankdata=sortdata(data,year,nums)
    return alt.Chart(rankdata).mark_circle().encode(
    alt.X(x, scale=alt.Scale(zero=False)),
    alt.Y(y, scale=alt.Scale(zero=False, padding=1)),
    color='rank',
    size=size
).properties(
    height=600,
    width=800
)


