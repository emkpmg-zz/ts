# Predict passengers' survival on the Titanic Ship

The Titanic collided with an iceberg on its first voyage and sank on April 15, 1912. Owning to insufficient lifeboats for the passengers and crew, about 1502 of its 2224 total passengers died. This tragic incident threw the international community into shock, leading to establishment and enforcement of safer ship regulations. This project makes analyses of the passengers who survived. In terms of gender, social class or age.

Two datasets are included:
-The training set (train.csv): used to build your machine learning models
-The test set (test.csv): used to assess how well your model performs on unseen data (Data other than what was used to train the model)

# Variable Definition

Survival (survival) -- Did Passenger Survive the Shipwreck?

0 = No

1 = Yes 


Ticket class (pclass) -- class of ticket passenger holds. This variable also describes socio-economic status

1 = 1st = Upper Class 

2 = 2nd = Middle Class 

3 = 3rd = Lower Class 


Sex (sex) -- Passenger's gender


Age (Age) -- Passenger's age in years. age: If age < 1, it will be estimated as a fraction. e.g xx.5


sibsp -- Number of passenger's siblings(brother, sister, stepbrother, stepsister) or spouses(husband or wife) on board


parch -- Number of parents(mother or father), children(daughter, son, stepdaughter, stepson) or relatives on board. Children traveling with Nannies had a null value for this variable.


ticket -- Passenger's ticket number


fare -- Cost of passenger's fare


cabin -- Cabin number 


embarked -- Port of Embarkation 

C = Cherbourg

Q = Queenstown

S = Southampton
