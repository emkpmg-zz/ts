# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:05:03 2019

@author: PIANDT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline

# 01 Data exploration: Load training or test data with pd.read_csv, print sample to see how data looks
trainingData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

#view sampls of training or test data
trainingData.sample(5)
testData.sample(5)

# 02 Data Visualization
#embarked -- Port of Embarkation C = Cherbourg Q = Queenstown S = Southampton
#sns.barplot -- central tendency of a numeric variable
v1 = sns.barplot(x="Embarked", y="Survived", hue="Sex", data=trainingData);
v1.set(xlabel='Port of Embarkation', ylabel='Survival Tendency')
plt.show()

v2 = sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=trainingData,
              palette={"male": "purple", "female": "turquoise"},
              markers=["*", "o"], linestyles=["-", "--"]);
v2.set(xlabel='Socio-economic status', ylabel='Survival Tendency')
plt.show()
#Heatmap correlation can help select the most important variables for our learning model
c = trainingData.corr()
sns.heatmap(c)

#custom heatmap fxn -- optional
#def heatMap( df ):
#    _ , a = plt.subplots( figsize =( 12 , 10 ) )
#    cm = sns.diverging_palette( 220 , 10 , as_cmap = True )
#    _ = sns.heatmap(c, cmap = cm, square=True, cbar_kws={ 'shrink' : .9 }, 
#        ax=a, annot = True, annot_kws = { 'fontsize' : 12 })
#heatMap(trainingData)
#Important variables in relation to survival: socio-economic class
# Others: socio-economic class&Age, socio-economic class&fare

# relationship between age and Survival grouped by sex
ageSurvive = sns.FacetGrid(trainingData , hue= 'Survived' , aspect=3 , row = 'Sex' )
ageSurvive.map( sns.kdeplot , 'Age' , shade= True )
ageSurvive.set( xlim=( 0 , trainingData[ 'Age' ].max() ) )
ageSurvive.add_legend()

# survival for of males and females according to age. Significant difference
#in curves is a sign that age is a relevant variable for predicting survival(target) 