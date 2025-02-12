import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# giving column names
names=["Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]

df=pd.read_csv("./vinedataset/wine.data",header=None,names=names)

# filling null values
df["Malic acid"]=df["Malic acid"].fillna(df["Malic acid"].mean())
df["Color intensity"]=df["Color intensity"].fillna(df["Color intensity"].mean())
df["Ash"]=df["Ash"].fillna(df["Ash"].mean())
df["Proline"]=df["Proline"].fillna(df["Proline"].mean())

# removing outliers
for names in df.columns:
    q1=df[names].quantile(0.25)
    q2=df[names].quantile(0.75)
    iqr=q2-q1
    alt=q1-1.5*iqr
    ust=q2+1.5*iqr
    df=df[(df[names]>=alt)&(df[names]<=ust)]

# training data

# scaler=MinMaxScaler()
# df=scaler.fit_transform(df)
# scaling data using RobustScaler
scaler = RobustScaler()
df = scaler.fit_transform(df)

# splitting data into train and test sets
x=df[:,1:]
y=df[:,0:1]
x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.3, random_state=42)
models={
    "Linear Regression":LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=42),

}
for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    accuracy=np.mean((y_pred-y_test)**2)
    print(f"{name} {accuracy*100:.2f}%")

#giving column names
# names=["Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
#
# df=pd.read_csv("./vinedataset/wine.data",header=None,names=names)
# #filling null values
# df["Malic acid"]=df["Malic acid"].fillna(df["Malic acid"].mean())
# df["Color intensity"]=df["Color intensity"].fillna(df["Color intensity"].mean())
# df["Ash"]=df["Ash"].fillna(df["Ash"].mean())
# df["Proline"]=df["Proline"].fillna(df["Proline"].mean())
#
# # print(df.isnull().sum()) #checking null values
# # #removing outliers
# for names in df.columns:
#     q1=df[names].quantile(0.25)
#     q2=df[names].quantile(0.75)
#     iqr=q2-q1
#     alt=q1-1.5*iqr
#     ust=q2+1.5*iqr
#     df=df[(df[names]>=alt)&(df[names]<=ust)]
#
# #training data
# scaler=RobustScaler()
# df=scaler.fit_transform(df)
# x=df.drop("Alcohol",axis=1)
# y=df["Alcohol"]
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.35,random_state=43)
#
# models={
#     "Logistic Regression":LogisticRegression(),
#     "Decision Tree":DecisionTreeClassifier(),
#     "Random Forest":RandomForestClassifier()
# }
# for name,model in models.items():
#     model.fit(x_train,y_train)
#     y_pred=model.predict(x_test)
#     accuracy=np.mean(y_pred==y_test)
#     print(name,accuracy*100)