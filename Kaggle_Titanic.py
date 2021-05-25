# Titanic, random forest.

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#read data, put it into numbers, fill nan
df = pd.read_csv('train.csv')
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(df.head())
df = df.replace({'male': 1, 'female': 2})
df = df.replace({'S': 1, 'C': 2, 'Q': 3}) #this is in another line for cleanliness only 
for column in df.columns:
	value = df[column].value_counts().max() #find most common value
	df[column] = df[column].fillna(value) #fill nan with that

#make test, train samples
X = df.iloc[:, 1:8].values
y = df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#run through random forest
classifier = RandomForestClassifier(n_estimators=200, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# # print out some useless info
# print(confusion_matrix(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))

#run the test data
df_test = pd.read_csv('test.csv')
data = pd.DataFrame(df_test['PassengerId'])
df_test = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df_test = df_test.replace({'male': 1, 'female': 2})
df_test = df_test.replace({'S': 1, 'C': 2, 'Q': 3}) #this is in another line for cleanliness only 
for column in df_test.columns:
	value = df_test[column].value_counts().max() #find most common value
	df_test[column] = df_test[column].fillna(value) #fill nan with that

X = df_test.values
data['Survived'] = classifier.predict(X)
data.to_csv('TitanicSubmission.csv', index=False)