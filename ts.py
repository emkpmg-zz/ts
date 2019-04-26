# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:05:03 2019

@author: PIANDT
"""
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
#%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

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
#    _ , a = plt.subplots( figsize =( 10 , 8 ) )
#    cm = sns.diverging_palette( 220 , 10 , as_cmap = True )
#    _ = sns.heatmap(c, cmap = cm, square=True, cbar_kws={ 'shrink' : .9 }, 
#        ax=a, annot = True, annot_kws = { 'fontsize' : 14 })
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

#Grouping family size into categories with 'Parch' and 'Sibsp' variables
#create dataframe for family category called famCat
famCategory = pd.DataFrame()
famCat = pd.DataFrame()
# introducing a new feature : the size of families (including the passenger)
famCat[ 'FamilySize' ] = bothDatasets[ 'Parch' ] + bothDatasets[ 'SibSp' ] + 1
# introducing other features based on the family size
famCat[ 'Family_Single' ] = famCat[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
famCat[ 'Family_Small' ]  = famCat[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
famCat[ 'Family_Large' ]  = famCat[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
#view family category
famCat.head()

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
ageNfare = pd.DataFrame()

# Substitute missing Age values with mean Age
ageNfare[ 'Age' ] = bothDatasets.Age.fillna( bothDatasets.Age.mean() )

# Substitute missing Fare values with mean Fare
ageNfare[ 'Fare' ] = bothDatasets.Fare.fillna( bothDatasets.Fare.mean() )

#compare original data sample to imputed sample to observe populated missing values
ageNfare.sample(20)
bothDatasets.describe()

# 02 VARIABLE SYNTHESIS
#Titles of passengers' names can help give us ideas about their social status.

#Create passenger ID as part of the chosen vars: for indexing or identification 
pid = bothDatasets['PassengerId']
print(pid)


#Extracting titles from names we create a dataframe for titles , pTitle
pTitle = pd.DataFrame()
# we extract the title from each name
pTitle[ 'Title' ] = bothDatasets[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

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
pTitle[ 'Title' ] = pTitle.Title.map( Title_Dictionary )
pTitle = pd.get_dummies( pTitle.Title )
#title = pd.concat( [ title , titles_dummies ] , axis = 1 )
#view title data
pTitle.head()

#Cabin category can be obtained from cabin number
#create a cabin category dataframe
cabinCat = pd.DataFrame()
# Missing values for cabin number to be replaced with 'NA' - Not Available/ Non Applicable
cabinCat[ 'Cabin' ] = bothDatasets.Cabin.fillna( 'NA' )
# map cabin values to corresponding cabin letter
cabinCat[ 'Cabin' ] = cabinCat[ 'Cabin' ].map( lambda c : c[0] )
# dummy encoding
cabinCat = pd.get_dummies( cabinCat['Cabin'] , prefix = 'Cabin' )
#view cabin data
cabinCat.head()

# 03 VARIABLE SELECTION AND DATA SPLITING
#combine Chosen Variables: cabinCat, ageNfare, embarked, famCat, Sex, ticket, pclass
chosenVars = pd.concat([ageNfare, embarked, cabinCat, sex, pid], axis = 1)
print(chosenVars.describe())

#Data split: Training, Validation and Test sets creation post cleaning
# Create all datasets that are necessary to train, validate and test models
#X - input features and Y - output / labels
trainX = chosenVars[ 0:891 ]
trainY = trainingData.Survived
testX = chosenVars[ 891: ]
trainX , validX , trainY , validY = train_test_split( trainX , trainY , train_size = .7 )
print(chosenVars.shape , trainX.shape , validX.shape , trainY.shape , validY.shape , testX.shape)

#observe size of training, test and validation sets.

# Time to evaluate the most important variable for our model predictopns
# i.e which variables will make the most impact on our prediction model ?
    
tree = DecisionTreeClassifier( random_state = 99 )
tree.fit(trainX, trainY )
varRelevance = pd.DataFrame( tree.feature_importances_  , columns = [ 'Relevance' ] , index = trainX.columns)
varRelevance = varRelevance.sort_values( [ 'Relevance' ] , ascending = True )
varRelevance[ : 10 ].plot( kind = 'barh' )
print (tree.score( trainX , trainY ))


# 04 SELECTION OF A CLASSIFICATION MODEL
#Some ML models are Support Vector Machines, Gradient Boosting Classifier, Random Forests Model, K-nearest neighbors, Gaussian Naive Bayes and Logistic Regression.
# logisic regression model will be used for this classification
#Reason being, our output is a binary classification. 0 (did not survive) or 1 (survived)

#assign model to variable 'predictionModel'
predictionModel = LogisticRegression()

#fit model to training dataset
predictionModel.fit( trainX , trainY )

#Evaluate the accuracy or model or how well the model performs of training data
#This is done by comparing accuracy scores on both training and test set
print (predictionModel.score( trainX , trainY ) , predictionModel.score( validX , validY ))

#select relevant varibles automatically for model and visualize
rfecv = RFECV( estimator = predictionModel , step = 1 , cv = StratifiedKFold( trainY , 2 ) , scoring = 'accuracy' )
rfecv.fit( trainX , trainY )

print (rfecv.score( trainX , trainY ) , rfecv.score( validX , validY ))
print( "Ideal input features necessary for this model are : %d" % rfecv.n_features_ )

# Visualize ideal input features and validation test scores
plt.figure()
plt.xlabel( "Ideal Input Features" )
plt.ylabel( "Cross validation (CV)" )
plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )
plt.show()

#note that the feature selection can change eact time you run your code.
# The best number of features gives the highest accuracy for the model (e.g Logistic Regression)
# 4 gave an accuracy if 80.1

#EVALUATING ALL MODELS TO SEE WHICH ONES WORK BEST

# Logistic Regression
lr = LogisticRegression()
lr.fit(trainX , trainY)
survivePredictlr = lr.predict(testX)
lrAccuracy = round(lr.score(trainX, trainY) * 100, 2)
print('\n\nLogistic Regression Accuracy is   ', lrAccuracy)
print('\n\nLogistic Regression Survival Prediction on test dataset')
print(survivePredictlr, '\n\n')

#WE CAN FIND HIGHLY CORRELATED FEATURES
corrVar = pd.DataFrame(trainX.columns.delete(0))
corrVar.columns = ['Input Feature']
corrVar["Correlation"] = pd.Series(lr.coef_[0])
corrVar.sort_values(by='Correlation', ascending=False)
#input features will be printed in order of correlation relevance. Sex had the highest correlation
print(corrVar)


# Evaluating Support Vector Machines
svm = SVC()
svm.fit(trainX , trainY)
survivePredictsvm = svm.predict(testX)
svmAccuracy = round(svm.score(trainX , trainY) * 100, 2)
print('\n\nSupport Vector Machine Accuracy is   ', svmAccuracy)
print('\n\nSupport Vector Machine Survival Prediction on test dataset')
print(survivePredictsvm)
# You may observe that the SVM is predicting with a much higher accuracy than the Logistic Regression

# Evaluating k-Nearest Neighbors algorithm (k-NN)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(trainX , trainY)
survivePredictknn = knn.predict(testX)
knnAccuracy = round(knn.score(trainX , trainY) * 100, 2)
print('\n\nk-Nearest Neighbors algorithm Accuracy is   ', knnAccuracy)
print('\n\nk-Nearest Neighbors Survival Prediction on test dataset')
print(survivePredictknn)

# Evaluating Gaussian Naive Bayes
gauss = GaussianNB()
gauss.fit(trainX , trainY)
survivePredictgauss = gauss.predict(testX)
gaussAccuracy = round(gauss.score(trainX , trainY) * 100, 2)
print('\n\nGaussian Naive Bayes Accuracy is   ', gaussAccuracy)
print('\n\nGaussian Naive Bayes Survival Prediction on test dataset')
print(survivePredictgauss)


# Evaluating Perceptron model
percep = Perceptron()
percep.fit(trainX , trainY)
survivePredictpercep = percep.predict(testX)
percepAccuracy = round(percep.score(trainX , trainY) * 100, 2)
print('\n\nPerceptron model Accuracy is   ', percepAccuracy)
print('\n\nPerceptron model Survival Prediction on test dataset')
print(survivePredictpercep)


# Evaluating Linear SVC model
lsvc = LinearSVC()
lsvc.fit(trainX , trainY)
survivePredictlsvc = lsvc.predict(testX)
lsvcAccuracy = round(lsvc.score(trainX , trainY) * 100, 2)
print('\n\nLinear SVC model Accuracy is   ', lsvcAccuracy)
print('\n\nLinear SVC Survival Prediction on test dataset')
print(survivePredictlsvc)


# Evaluating Stochastic Gradient Descent model
sgd = SGDClassifier()
sgd.fit(trainX , trainY)
survivePredictsgd = sgd.predict(testX)
sgdAccuracy = round(sgd.score(trainX , trainY) * 100, 2)
print('\n\nStochastic Gradient Descent Accuracy is   ', sgdAccuracy)
print('\n\nStochastic Gradient Descent Survival Prediction on test dataset')
print(survivePredictsgd)


# Evaluating Decision Tree
dt = DecisionTreeClassifier()
dt.fit(trainX , trainY)
survivePredictdt = dt.predict(testX)
dtAccuracy = round(dt.score(trainX , trainY) * 100, 2)
print('\n\nDecision Tree Accuracy is   ', dtAccuracy)
print('\n\nDecision Tree Survival Prediction on test dataset')
print(survivePredictdt)


# Evaluating Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(trainX , trainY)
survivePredictrf = rf.predict(testX)
rf.score(trainX , trainY)
rfAccuracy = round(rf.score(trainX , trainY) * 100, 2)
print('\n\nRandom Forest Accuracy is   ', rfAccuracy)
print('\n\nRandom Forest Survival Prediction on test dataset')
print(survivePredictrf)

# HOW DO WE CHOOSE THE BEST MODEL ? -- One with highest accuracy !
allModels = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [svmAccuracy, knnAccuracy, lrAccuracy, 
              rfAccuracy, gaussAccuracy, percepAccuracy, 
              sgdAccuracy, lsvcAccuracy, dtAccuracy]})
allModels.sort_values(by='Score', ascending=False)

print(allModels)

#Generate an output for prediction
#since decision tree and random forest have the highest accuracies,#
#I choose to use the predictions from Decision Tree: 
predictionOutput = pd.DataFrame({
        "PassengerId": testX["PassengerId"],
        "Survived": survivePredictdt
    })

print(predictionOutput)




