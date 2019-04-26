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

# 01 DATA LOADING, EXPLORATION AND VISUALIZATION
#Load training or test data with pd.read_csv, print sample to see how data looks
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
# Difference in curves is a sign that age is a relevant variable for predicting survival(target)

# 02 DATA TRANSFORMATION AND MISSING VALUES

#Categorical variables (Embarked, Pclass and Sex)
#Necessary to convert to numeric value to #ensure easy data processing by model algorithms.
#This is done by creating dummy variables of numeric values for every categorical variable

#combine both training and test datasets
bothDatasets = trainingData.append( testData , ignore_index = True )

#select 891 rows out of bothDatasets (891 is the count of the bigger set - training set)
#selectedData = bothDatasets[ :891 ]

# Transform var Sex into 0 and 1 (Both for Training and Test Datasets). This is boolean, thus we have to hypothesize that gender is for
# instance female. So if  true, Sex = 1, else Sex = 0
sex = pd.Series( np.where( bothDatasets.Sex == 'female' , 1 , 0 ) , name = 'Sex' )
#view top rows for var Sex
sex.head()

# Create a new variable for every unique value of Embarked
embarked = pd.get_dummies( bothDatasets.Embarked , prefix='Embarked' )
#view top rows for var embarked
embarked.head()

#Missing Variables
#Most alghorims don't expect null values. Thus, missing values for variables can be polulated with an average of that variable's values as observed in the training set.

# Create an empty dataset to hold the populated missing values for Age and Fare
imputed = pd.DataFrame()

# Substitute missing Age values with mean Age
imputed[ 'Age' ] = bothDatasets.Age.fillna( bothDatasets.Age.mean() )

# Substitute missing Fare values with mean Fare
imputed[ 'Fare' ] = bothDatasets.Fare.fillna( bothDatasets.Fare.mean() )

#compare original data sample to imputed sample to observe populated missing values
imputed.sample(20)
bothDatasets.sample(20)

# 02 VARIABLE SYNTHESIS
#Titles of passengers' names can help give us ideas about their social status.

#Extracting titles from names we create a dataframe for titles 
title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = bothDatasets[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

# dictionary with all possible titles
Title_Dictionary = {
                    "Capt":       "Officer", "Col":        "Officer", "Major":      "Officer",
                    "Jonkheer":   "Royal Family", "Don":        "Royal Family", "Sir" :       "Royal Family",
                    "Dr":         "Officer", "Rev":        "Officer", "the Countess":"Royal Family",
                    "Dona":       "Royal Family", "Mme":        "Mrs", "Mlle":       "Miss",
                    "Ms":         "Mrs", "Mr" :            "Mr", "Mrs" :       "Mrs",
                    "Miss" :      "Miss", "Master" :    "Master", "Lady" :      "Royal Family"
                    }

# mapping each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )
#title = pd.concat( [ title , titles_dummies ] , axis = 1 )
#view title data
title.head()

#Cabin category can be obtained from cabin number
#create a cabin category dataframe
cabin = pd.DataFrame()
# Missing values for cabin number to be replaced with 'NA' - Not Available/ Non Applicable
cabin[ 'Cabin' ] = bothDatasets.Cabin.fillna( 'NA' )
# map cabin values to corresponding cabin letter
cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )
# dummy encoding
cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )
#view cabin data
cabin.head()


#Get class of ticket from the ticket number
#function to extracts ticket prefix, returns 'PNA' if no prefix found or ticket is made of only numbers
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'PNA'
ticket = pd.DataFrame()

# Get dummy variables from tickets:
ticket[ 'Ticket' ] = bothDatasets[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )
ticket.shape
ticket.head()

