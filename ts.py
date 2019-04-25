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
#sns.barplot -- 
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=trainingData);