#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 01:15:35 2018

@author: jinqingyuan
"""

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

text = 'GDP,HIV,Population, Schooling, Mortality,infant,Alcohol,expenditure%,Measles,BMI,15belowdeaths,Polio,ExpenditureTotal,Diphtheria,IncomeResources,GDP,YoungThinness'

def visual_wordcloud(text):
    '''
    This part we achieve showing all features randomly by wordcloud.
    This fuction will return one graph consisting all features' name.
    Input:
    text(datatype:str):Name of all features
    Output:
    Graph:'visual_wordcould.jpg'
    '''
    assert isinstance(text,str)
    
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 300 ** 2
    mask = 255 * mask.astype(int)
    wc = WordCloud(background_color='white')
    wc.generate(text)
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    plt.savefig('visual_wordcould.jpg',dpi=500)
    plt.show()
    plt.close()
    
    return
