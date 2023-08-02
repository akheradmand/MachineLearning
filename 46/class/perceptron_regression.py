import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import time

data=pd.read_csv("weight-height.csv")

X=data["Height"].values
Y=data["Weight"].values

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.95,shuffle=True)

X_train=X_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)
Y_test=Y_test.reshape(-1,1)
# print(X_train)
# print(Y_train)

fig,(ax1,ax2)=plt.subplots(1,2)

# Training
W = np.random.rand(1,1)
b = np.random.rand(1,1)

learninig_rate_W=0.0001
learninig_rate_b=0.1
# print(W)

losses = []

for j in range(50):
    for i in range(X_train.shape[0]):
        x=X_train[i]
        y=Y_train[i]
        y_pred=W*x+b
        error=y-y_pred

        # SGD update
        W = W + (error * x * learninig_rate_W)
        b = b + (error * learninig_rate_b)
        # print(W)
        # time.sleep(0.5)

        # mae
        loss = np.mean(np.abs(error))
        losses.append(loss)

        Y_pred=X_train*W + b
        ax1.clear()
        ax1.scatter(X_train,Y_train,color="blue")
        ax1.plot(X_train,Y_pred,color='red')
        # ax1.title("data and fitted line")

        ax2.clear()
        ax2.plot(losses)
        # ax2.title("loss")
        
        plt.pause(0.01)