# Artificial Neural Network

# Part 1:
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer([('One_hot_encoder',
                                OneHotEncoder(),[1])],
                              remainder='passthrough')
X = np.array(onehotencoder.fit_transform(X))
X = X[:,1:]


# Spliting the dataset into training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,
                                                    random_state = 0)


# Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)




# Part 2: Creating the ANN model

# Importing the Deep learning libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout

# Initalizing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, input_dim=11, kernel_initializer='uniform', 
                     activation = 'relu'))
#classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation = 'relu'))
#classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy',
                   metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train,y_train, batch_size=100, epochs=100) 





# Part 3: Making the prediction and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Applying the threshold to filter the predicted output
y_pred = (y_pred>0.5)


# Computing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Prdicting a single new observation
"""Predicting if the customer with the following information will leave the bank? 
Geography: France
Credit score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
# of Products: 2
Has Credit Card: Yes
Is Active member: Yes
Estimated Salary: 50000
"""
new_predict = classifier.predict(sc.fit_transform(np.array([[0,0, 600, 1, 
                                                             40, 3, 60000,
                                                             2, 1, 1, 
                                                             50000]])))
new_predict = new_predict>0.5




# Part 4: Evaluating, Improving and Tuineing the ANN model

# Initalizating the K-Fold and Evalutating the model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, input_dim=11, kernel_initializer='uniform', 
                     activation = 'relu'))
    classifier.add(Dense(6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss = 'binary_crossentropy',
                   metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,
                               batch_size = 10,
                               epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, 
                             y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()


# Improving regularization to reduce overfitting if needed


# Importing the Deep learning libraries using parametes tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, input_dim=11, kernel_initializer='uniform', 
                     activation = 'relu'))
    classifier.add(Dense(6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss = 'binary_crossentropy',
                   metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,
                               batch_size = 10,
                               epochs = 100)
parametes = {'batch_size':[25, 32],
             'epochs':[100,500],
             'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parametes,
                           scoring = 'accuracy',
                           cv = 10)

grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_







