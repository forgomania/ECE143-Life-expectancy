# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from altair import Chart, load_dataset
import altair as alt

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



def showScatter(data,x,size=None,y='Life expectancy ',year=2000, nums=20,height=600,width=800,title="Default_Title"):
    """
    show scatter plot of features with y( default to be life expectancy), with option to change size according to another feature.
    
    params:
        data, originaldata, pd.dataframe
        y, y-axis feature, str
        x, x-axis feature, str
        size, the size we want set for x feature
        nums, the number of countries we want to present, int
        year, the yeart we want to extract data and show from, int
        height, the height of the graph we want to plot, int
        width, the width of the graph we want to plot, int
    """
    
    assert isinstance(data,pd.core.frame.DataFrame)
    assert isinstance(y,str)
    assert isinstance(nums,int)
    assert isinstance(year,int) and data.Year.min() <= year <= data.Year.max()  
    assert isinstance(height,int) and isinstance(width,int)
    assert isinstance(x,str)
    
    rankdata=sortdata(data,year,nums)
    
    return alt.Chart(rankdata,title=title).mark_circle().encode(
    alt.X(x, scale=alt.Scale(zero=False)),
    alt.Y(y, scale=alt.Scale(zero=False, padding=1)),
    color='rank',
    size=size).properties(height=height,width=width).configure_axis(titleFontSize=15)
    

def showBar_alt(data,y='Life expectancy ',nums=10,year=2000,height=600,width=800,title=None):
    """
    show bar plot of a feature in given year with top nums countries
    
    params:
        data, originaldata, pd.dataframe
        y, y-axis, str
        nums, the number of countries we want to present, int
        year, the yeart we want to extract data and show from, int
        height, the height of the graph we want to plot, int
        width, the width of the graph we want to plot, int
    """
    
    assert isinstance(data,pd.core.frame.DataFrame)
    assert isinstance(y,str)
    assert isinstance(nums,int)
    assert isinstance(year,int) and data.Year.min() <= year <= data.Year.max()  
    assert isinstance(height,int) and isinstance(width,int)
    
    
    
    newdata=sortdata(data,year,nums)
    
    return alt.Chart(newdata,title=title).mark_bar().encode(
    x=alt.X('Life expectancy ', axis=alt.Axis(title='Life Expectancy')),
    y=alt.Y('Country', sort=alt.EncodingSortField(field='Life Expectancy ', op='sum', order='ascending')),
    color='rank',
    ).properties(
    height=height,
    width=width
    ).configure_axis(titleFontSize=15)

    

