import pandas as pd 
import numpy as np 

data = pd.read_csv('glass.csv')

x = data.drop('Type',axis = 1)
y = data['Type']

#train test split
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y)


#shuffling the trian data for avoiding the probability of overfitting
from sklearn.utils import shuffle
xtrain, ytrain= shuffle(xtrain,ytrain, random_state=0)

#scaling the data for better processing
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(xtrain)
x_test = ss.transform(xtest)


#creating the model and training teh data
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion = 'entropy', n_estimators=300)
clf.fit(x_train,ytrain)

# from sklearn.svm import SVC
# clf=SVC(gamma='auto')				
# clf.fit(x_train, ytrain)

#predidtion
y_pred = clf.predict(x_test)

#accuracy of the traning
print('training accuracy : ',clf.score(x_train, ytrain),'%')
print('testing accuracy : ',clf.score(x_test, ytest),'%')

#saving the model using the joblib
import joblib
filename = 'glass.sav'
joblib.dump(clf, filename)
