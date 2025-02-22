# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:07:09 2018

@author: Weinan(Eric) Li. w3li@ucsd.edu
"""

#requires pygal,pygal_maps_world,pandas,numpy

import pygal.maps.world
from IPython.display import SVG
from Project.Data_Cleaning import clean_data
import pygal.style
from pygal.style import Style
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
def display(data,feature,year):
    '''
    Takes a CleanData object "data", a "feature" from this dataset other than Country
    or Year, and a specific year. On a world map, draws how this feature is distributed
    in the given year. The year should not be outside the range of the dataset.
    
    param: data     type: CleanData object
    param: feature  type: str
    param: year     type: int
    '''
    from pygal_maps_world.i18n import COUNTRIES
    
    assert isinstance(data, clean_data.CleanData)
    
    assert isinstance(feature,str)
    valid_features = list(data.modified.columns)
    valid_features.remove('Country')
    valid_features.remove('Year')
    assert feature in valid_features
    
    assert isinstance(year,int)
    modified = data.modified
    assert modified.Year.min() <= year <= modified.Year.max()   
    
    #match the country codes
    countries={value:key for key, value in COUNTRIES.items()}
    countries['United States of America']='us' # there're more needs to manually match
    countries['United States of America']='us' # there're more needs to manually match
    countries['United Kingdom of Great Britain and Northern Ireland']='gb'
    countries['Bolivia (Plurinational State of)']='bo'
    countries["Côte d'Ivoire"] = 'ci'
    countries['Cabo Verde']='cv'
    countries['Czechia']='cz'
    countries["Democratic People's Republic of Korea"]='kp'
    countries['Democratic Republic of the Congo']='cd'
    countries['Iran (Islamic Republic of)']='ir'
    countries['Libya']='ly'
    countries['Republic of Korea']='kr'
    countries['Republic of Moldova']='md'
    countries['The former Yugoslav republic of Macedonia']='mk'
    countries['United Republic of Tanzania']='tz'
    countries['Venezuela (Bolivarian Republic of)']='ve'
    
    
    display_data=dict()
    
    for i in range(modified.shape[0]):
        row=modified.loc[i]
        if row['Year']==year:
            countryname=row['Country']
            display_feature=row[feature]
            try:
                display_data[countries[countryname]]=display_feature
            except:
                pass
    
    #colors need to be adjusted for clearer display
    worldmap_chart = pygal.maps.world.World()
    worldmap_chart.title = '{0} in the year {1}'.format(feature,year)
    worldmap_chart.add('In {0}'.format(year), display_data)
    
    return SVG(worldmap_chart.render())



def displaymap(data,feature,year,color1=(255,100,100),color2=(1,200,200)):
    """
    Show feature of all countries in data on map.
    Custom style
    Use color1 and color2 as two extreme values, set each country's color based on gradient.
    
    input:
        data, panda.dataframe
        feature, str
        year, int, given year
    output:
        map
    """
    
    
    from pygal_maps_world.i18n import COUNTRIES
    
    assert isinstance(data, clean_data.CleanData)
    
    assert isinstance(feature,str)
    valid_features = list(data.modified.columns)
    valid_features.remove('Country')
    valid_features.remove('Year')
    assert feature in valid_features
    
    assert isinstance(year,int)
    modified = data.modified
    assert modified.Year.min() <= year <= modified.Year.max()   
    
    #match the country codes
    countries={value:key for key, value in COUNTRIES.items()}
    countries['United States of America']='us' # there're more needs to manually match
    countries['United States of America']='us' # there're more needs to manually match
    countries['United Kingdom of Great Britain and Northern Ireland']='gb'
    countries['Bolivia (Plurinational State of)']='bo'
    countries["Côte d'Ivoire"] = 'ci'
    countries['Cabo Verde']='cv'
    countries['Czechia']='cz'
    countries["Democratic People's Republic of Korea"]='kp'
    countries['Democratic Republic of the Congo']='cd'
    countries['Iran (Islamic Republic of)']='ir'
    countries['Libya']='ly'
    countries['Republic of Korea']='kr'
    countries['Republic of Moldova']='md'
    countries['The former Yugoslav republic of Macedonia']='mk'
    countries['United Republic of Tanzania']='tz'
    countries['Venezuela (Bolivarian Republic of)']='ve'
    
    
    
    modified = data.modified
    lifedata=dict()
    #brute force, needs perfection, only mean to show result
    for i in range(modified.shape[0]):
        row=modified.loc[i]
        if row['Year']==year:
            countryname=row['Country']
            target=row[feature]
            try:
                lifedata[countries[countryname]]=target
            except:
                #print(countryname)
                pass
        
    #use extreme values in 15 years to build color bar and determine colors of each country
    maxValue=max(modified['Life expectancy '])
    minValue=min(modified['Life expectancy '])
    
    
    lifedata=sorted(lifedata.items(), key=lambda d: d[1])
    
    #colors need to be adjusted for clearer display

    #fractionlist=list(np.linspace(0,1,len(lifedata)))
    #the colors shown on colorbar
    colors = plt.cm.RdYlGn(np.linspace(0,1.,129)) 
    colors=colors*255
    colors=colors.astype(int)
    colors[1][0:3]
    colorlist=list()
    for i in lifedata:
        index=int(128*(i[1]-minValue)/(maxValue-minValue))
    
        #rgb=blend_color(color1,color2,fraction)
        #hexa=decimal2hex(rgb)
        rgb=colors[index][0:3]
        hexa=decimal2hex(rgb)
        colorlist.append(hexa)
    #lifedata=lifedata.sort_index(by='Value')
    
    custom_style = Style(
  background='transparent',
  plot_background='transparent',
  foreground='#53E89B',
  foreground_strong='#53A0E8',
  foreground_subtle='#630C0D',
  opacity='.6',
  opacity_hover='.9',
  transition='400ms ease-in',
  colors=tuple(colorlist))
    
    worldmap_chart = pygal.maps.world.World(style=custom_style)
    worldmap_chart.title = feature+' in '+str(year)
    for i in lifedata:
        worldmap_chart.add("",{i[0]:i[1]})
    """
    worldmap_chart.add('<40', life40.to_dict()['Value'])
    worldmap_chart.add('<50', life50.to_dict()['Value'])
    worldmap_chart.add('<60', life60.to_dict()['Value'])
    worldmap_chart.add('<70', life70.to_dict()['Value'])
    worldmap_chart.add('<80', life80.to_dict()['Value'])
    worldmap_chart.add('<90', life90.to_dict()['Value'])
    """
    #worldmap_chart.add('<40', lifedata.to_dict()['Value'])
    #return colorlist
    return SVG(worldmap_chart.render())




def blend_color(color1, color2, f):
    """
    find gradual color between color1 and color2, f as fraction
    """
    assert isinstance(color1,tuple)
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    r = r1 + (r2 - r1) * f
    g = g1 + (g2 - g1) * f
    b = b1 + (b2 - b1) * f
    return r, g, b
def decimal2hex(color):
    """
    convert decimal rgb color to hex
    """
    ans='#'
    for i in color:
        if len(hex(int(i))[2:])<2:
            ans+='0'+hex(int(i))[2:]
        else:
            ans+=hex(int(i))[2:]
            
    return ans


def f(x, y):
    return x

def createBar():
    """
    use matplotlib to create a color bar to show on the map.
    Need to paste externally.
    """
    from matplotlib import pyplot as plt
    import numpy as np

    maxValue=max(modified['Life expectancy '])
    minValue=min(modified['Life expectancy '])
    n = 10
    x = np.linspace(minValue,maxValue,100)
    y = np.linspace(minValue,maxValue,100)
    X, Y = np.meshgrid(x, y)
    plt.imshow(f(X, Y), cmap='RdYlGn', origin='low')
    plt.colorbar()
     
    plt.xticks(())
    plt.yticks(())
    plt.show()