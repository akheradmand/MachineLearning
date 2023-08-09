import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

data=pd.read_csv("abalone.csv")

# data["Sex"]=data["Sex"].replace(["F","M","I"],[0,1,2])

X=data["Length"].values.reshape(-1,1)
Y=data["Height"].values.reshape(-1,1)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.8)

W=np.random.rand(1,1)
b=np.random.rand(1,1)
lr_W=0.0005
lr_b=0.1
epochs=2

per=Perceptron(W,b,lr_W,lr_b,epochs)
per.fit(X_train,Y_train)