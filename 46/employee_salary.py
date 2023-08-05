import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

X,Y,coef = make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, coef=True, random_state=1)

Y=Y.reshape(-1,1)

# Scale feature x (years of experience) to range 0-30
X = np.interp(X, (X.min(), X.max()), (0, 20))

# Scale target y (salary) to range 20000-150000 
Y = np.interp(Y, (Y.min(), Y.max()), (20000, 150000))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#training
W=np.random.rand(1,1)
b=np.random.rand(1,1)

lr_w=0.0001
lr_b=0.1

epochs=20

perceptron=Perceptron(W,b,lr_w,lr_b,epochs)
perceptron.fit(X_train,Y_train)

#predict with X_test
perceptron.predict(X_test)