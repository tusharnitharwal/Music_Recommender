import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df= pd.read_csv('music.csv')
X = df.drop(columns=['genre'])
y= df['genre']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
model= tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
score = accuracy_score(y_test, predictions)
predictions
# score