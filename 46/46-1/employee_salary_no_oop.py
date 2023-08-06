import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

X,Y,coef = make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, coef=True, random_state=0)

Y=Y.reshape(-1,1)

# Scale feature x (years of experience) to range 0-30
X = np.interp(X, (X.min(), X.max()), (0, 20))

# Scale target y (salary) to range 20000-150000 
Y = np.interp(Y, (Y.min(), Y.max()), (20000, 150000))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

fig,(ax1,ax2)=plt.subplots(1,2)

#training
W=np.random.rand(1,1)
b=np.random.rand(1,1)

learning_rate_w=0.0001
learning_rate_b=0.1

losses=[]

for n in range(50):
    for i in range(X_train.shape[0]):
        x=X_train[i]
        y=Y_train[i]
        y_pred=x*W+b
        
        error=y-y_pred

        W = W + error*x*learning_rate_w
        b = b + error*learning_rate_b

        loss=np.mean(np.abs(error))
        losses.append(loss)

        Y_pred=X_train*W+b

        ax1.clear()
        ax1.scatter(X_train,Y_train,color="blue")
        ax1.plot(X_train,Y_pred, color="red")

        ax2.clear()
        ax2.plot(losses)
        plt.pause(0.01)
