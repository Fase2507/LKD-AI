import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
names=["Target","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
df=pd.read_csv('./vinedataset/wine.data',header=None,names=names)

df=df.dropna()
# print(df.isnull().sum())



X=df.drop('Target',axis=1)
y=df['Target']
# print(X,y)

for name in names[1:]:
    q1=df[name].quantile(0.25)
    q3=df[name].quantile(0.75)
    iqr=q3-q1
    alt=q1-1.5*iqr
    ust=q3-1.5*iqr
    df=df[(df[name]>=alt)&(df[name]<=ust)]
scaler=RobustScaler()
X=scaler.fit_transform(X)
# print(X,y)

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=.3,random_state=42)
accuracy_values={}

models={
    "K-NN":KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier":RandomForestClassifier()
}

for name,model in models.items():
    model.fit(x_train,y_train)
    y_prediction = model.predict(x_test)
    acc=accuracy_score(y_test,y_prediction)
    accuracy_values[name]=acc
for name,acc in accuracy_values.items():
    print(f"Accuracy rate {name}: {acc*100:.2f}%")