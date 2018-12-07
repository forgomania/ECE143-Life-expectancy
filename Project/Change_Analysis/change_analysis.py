# -*- coding: utf-8 -*-
"""
Created on Sat Dec 1

@author: Wei Zhang

This module's purpose is to generate the results for Part 3 of our analysis.

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_results(df):

    assert isinstance(df,pd.DataFrame)
    
    pd.set_option('display.max_columns', None)   #to show one whole column 
       
    df_2015 = pd.DataFrame(columns=df.columns) # to save datas of 2015
    df_2000 = pd.DataFrame(columns=df.columns)#to save datas of 2000
    
    for i in range(0, len(df)):
        if df.iloc[i]['Year'] == 2015:
            df_2015.loc[df_2015.shape[0]+1] = df.iloc[i]
        elif df.iloc[i]['Year'] == 2000:
            df_2000.loc[df_2000.shape[0]+1] = df.iloc[i]
    # df_2015 = df_2015.set_index('Country',inplace=True, drop=True)    # set 'country' as index
    df_2015.drop(['Year', 'Status', 'Measles ','Alcohol','infant deaths','Polio','Diphtheria ',' thinness  1-19 years','percentage expenditure','Total expenditure',' BMI ','Population','under-five deaths ',' thinness 5-9 years'], axis=1, inplace=True)  # remove useless columns
    df_2015.set_index('Country', inplace=True, drop=True)               # set 'country 'as index
    df_2000.drop(['Year', 'Status', 'Measles ','Alcohol','infant deaths','Polio','Diphtheria ',' thinness  1-19 years','percentage expenditure','Total expenditure',' BMI ','Population','under-five deaths ',' thinness 5-9 years'], axis=1, inplace=True)   
    df_2000.set_index('Country', inplace=True, drop=True)
    df_dif15 = (df_2015-df_2000)/df_2000                 #  changing rate of features of all countries from 2000 to 2015
    df_dif15mean=df_dif15.mean()
    df_Pos = df_dif15.sort_values(by = 'Life expectancy ', ascending=False) # descendingly re-order df_dif15
    
    df_Pos = df_Pos[0:10]     # choose top 10 countries 
    Top10mean_fea=df_Pos.mean()
    draw = (df_Pos-df_dif15.mean())/df_dif15.mean()             
    draw.drop(['Life expectancy ', ], axis=1, inplace=True)   
    draw.plot(kind='bar')
    
    
    feas = list(df_2015.columns)
    
    '''Plot1:
    top10 countries VS all counries 
    in features' average changing rates
    
    '''
    plt.figure(figsize=(22,14))
    name_list = feas#xtick name
    name_list[-2]='Income'
    X = np.arange(len(feas))
    Y1=Top10mean_fea.tolist()
    Y2=df_dif15mean.tolist()
    Y2[3]=Y2[3]*(-1)
    #print(Y1)
    #print(Y2)
    plt.rcParams['font.size']=18
    plt.bar(X, Y1, alpha=0.8, width = 0.35, facecolor = 'green', label='Top 10 Countries', lw=2.3, tick_label=name_list)
    plt.bar(X+0.35, Y2, alpha=0.8, width = 0.35, facecolor = 'orange', label='All Countries', lw=2.3)
    plt.legend(loc="upper right",fontsize=25) 
    plt.hlines(0,-0.5,7)
    
    #add texts on/below the bar
    for x1,y1 in enumerate(Y1):
        if y1>0:
            plt.text(x1, y1, '%s' %round(y1,2),ha='center')
        else:
            plt.text(x1, y1-0.05, '%s' %round(y1,2),ha='center')
    for x2,y2 in enumerate(Y2):
        if y2>0:
            plt.text(x2+0.35, y2, '%s' %round(y2,2),ha='center')
        else:
            plt.text(x2+0.35, y2-0.05, '%s' %round(y2,2),ha='center')
    plt.ylabel('Changing Rate', size=30)
    plt.xlabel('Features', size=30)
    plt.xticks(rotation=40, size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.title("top10 countries VS all counries \n in features' average changing rates", fontsize=30)
    #plt.savefig('aaa new doublebar111.png', dpi=500, bbox_inches = 'tight')
    plt.show()
    
    
    '''Plot2:double bar chart
    top10 countries VS all counries 
    in features' average changing rates 
    (WITHOUT GDP)
    '''
    plt.figure(figsize=(22,15))
    plt.rcParams['font.size']=18
    name_list = Top10mean_fea.drop('GDP').index.tolist()
    name_list[-2]='Income'
    Y1_dropGDP = Top10mean_fea.drop('GDP').tolist()
    Y2_dropGDP  = df_dif15mean.drop('GDP').tolist()
    Y2_dropGDP[3]=(-1)*Y2_dropGDP[3]
    X = np.arange(len(name_list))
    plt.bar(X, Y1_dropGDP, alpha=0.8, width = 0.35, facecolor = 'green', label='Top 10 Countries', lw=2.3, tick_label=name_list)
    plt.bar(X+0.35, Y2_dropGDP, alpha=0.8, width = 0.35, facecolor = 'orange', label='All Countries', lw=2.3)
    plt.legend(loc="upper right",fontsize=25) 
    plt.hlines(0,-0.5,6)
    
    for x1,y1 in enumerate(Y1_dropGDP):
        if y1>0:
            plt.text(x1, y1+0.01, '%s' %round(y1,2),ha='center')
            plt.annotate('buttom',xy=(x1,-1),xytext=(x1,0),arrowprops=dict(facecolor='blue', shrink=0.05))
        else:
            plt.text(x1, y1-0.03, '%s' %round(y1,2),ha='center')
            plt.annotate('buttom',xy=(x1,-1),xytext=(x1,y1),arrowprops=dict(facecolor='blue', shrink=0.05))
    for x2,y2 in enumerate(Y2_dropGDP):
        if y2>0:
            plt.text(x2+0.35, y2+0.01, '%s' %round(y2,2),ha='center')
        else:
            plt.text(x2+0.35, y2-0.03, '%s' %round(y2,2),ha='center')
    plt.ylabel('Changing Rate', size=30)
    plt.xlabel('Features', size=30)
    plt.xticks(rotation=40, size=20)
    plt.yticks(size=20)
    plt.ylim(-0.9,0.5)
    plt.tight_layout()
    plt.title("top10 countries VS all counries \n in features' average changing rates \n (WITHOUT GDP)", fontsize=30)
    #plt.savefig('2new doublebar no GDP.png', dpi=800,bbox_inches = 'tight')
    plt.show()
    
    
    '''Plot 3:Pie Chart
    distribution 
    of the life expectancy changing rate
    '''
    lst_LE=df_dif15['Life expectancy '].tolist()
    count_LE=[0,0,0,0,0]
    
    for i in lst_LE:
       if i >=0.2:
           count_LE[0]=count_LE[0]+1
       elif i>=0.1:
           count_LE[1]=count_LE[1]+1
       elif i>=0.05:
           count_LE[2]=count_LE[2]+1
       elif i>=0:
           count_LE[3]=count_LE[3]+1
       else:count_LE[4]=count_LE[4]+1
    
    plt.rcParams['font.size']=20
    plt.figure(figsize=(22,15)) 
    labels = [u'>20%',u'10%-20%',u'5%-10%',u'0-5%',u'<0'] 
    sizes = count_LE 
    
    colors = ['lightcoral','wheat','darkseagreen','lightblue','mediumpurple']
    explode = (0.02,0.02,0.02,0.02,0.02) 
    patches,text1,text2 = plt.pie(sizes,
                         explode=explode,
                         # labels=labels,
                         colors=colors,
                         autopct = '%3.2f%%', 
                         shadow = False, 
                         startangle =90,
                         pctdistance = 0.6) 
    
    plt.axis('equal')
    
    plt.legend(labels,bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.,fontsize=25)
    plt.title("'Life Expectancy' Changing Rate Ditribution", fontsize=30)
    plt.tight_layout()
    #plt.savefig('2new pie chart .png', dpi=500)
    plt.show()
    
    
    '''Plot 4:bar chart
    Top 10 Countries in life expectancy changing rate
    '''
    plt.figure(figsize=(15,10))
    name_list = df_Pos.index.tolist()
    num_list = df_Pos['Life expectancy ']
    plt.barh(range(len(num_list)), num_list,tick_label = name_list,color='red',alpha=0.5)
    plt.xlabel('Life Expectancy Changing Rate',size=25)
    
    plt.ylabel('TOP 10 Countries',size=25)
    plt.title('Top 10 countries in life expectancy changing rate ',fontsize=25 )
    plt.tight_layout()
    
    #plt.savefig('2Top10 Countries.png', dpi=500,bbox_inches = 'tight')
    plt.show()
    
    
    '''Plot 5:Bar chart
    Top10 Vs Overall
    '''
    Y1=Top10mean_fea.tolist()
    Y2=df_dif15mean.tolist()
    
    name_list = feas
    del name_list[0]
    
    X2 = np.arange(len(name_list))
    Y=list(map(lambda a,b:a/b,Y1,Y2))
    Y[3]=-1*Y[3]
    del Y[0]
    d = dict(zip(name_list, Y))
    d_sorted=sorted(d.items(),key=lambda item:item[1],reverse=True)
    name_list_sorted=[]
    Y_sorted=[]
    
    for i in d_sorted:
        name_list_sorted.append(i[0])
        Y_sorted.append(i[1])
        
    plt.figure(figsize=(15,12))
    plt.rcParams['font.size']=12
    plt.bar(X2,Y_sorted, alpha=0.6,  facecolor = 'seagreen',lw=1 ,tick_label=name_list_sorted)
    plt.hlines(0,-0.5,6)
    plt.ylabel('Top10 / Overall',size=25)
    plt.xlabel('Features',size=25)
    plt.xticks( rotation=40,size=18)
    plt.yticks( size=18)
    plt.tight_layout()
    plt.title('')
    plt.tight_layout()
    #plt.savefig('2top10overall bar.png', dpi=500)
    plt.show();

