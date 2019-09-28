# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:23:02 2019

@author: stomioka
"""

from matplotlib import pyplot as plt
import seaborn as sns

def corr_matrix(df):
    '''generates correlation matrix
    '''
    corr = df.corr() 
    fig=plt.figure(figsize=(12, 10))
    fig.add_subplot(111)
    plt.title('Tumor Correlation',fontsize=20)
    sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
                cmap='viridis'
                , vmax=1.0
                , vmin=-1.0
                , linewidths=0.1
                , annot=True
                , annot_kws={"size": 8}
                , square=True)


def plot_model_performance(df, t):
    '''plot model performance
    t --- is for leading text for title
    '''
    fig, ax = plt.subplots(figsize=(11.7,8.27))
    sns.set_style('white')
    g=sns.boxplot(x='model name', y='accuracy', data=df)
    sns.stripplot(x='model name', y='accuracy', data=df, 
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    fig.suptitle(t+' accuracy of each model', fontsize=25)
    plt.xticks(rotation=45, fontsize=14)
    g.set_xlabel("Model Name",fontsize=14)
    g.set_ylabel("accuracy",fontsize=14)
    g.tick_params(labelsize=14)
    plt.show()