import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from perceptron import Perceptron

boston = load_boston()

boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['TARGET'] = pd.Series(boston.target)

X=np.array((boston_df["TAX"],boston_df["RAD"])).T
Y=np.array((boston_df["TARGET"])).reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size=0.2)

W=np.random.rand(1,2)
b=np.random.rand(1,2)
lr_W=0.0005
lr_b=0.1
epochs=2

per=Perceptron(W,b,lr_W,lr_b,epochs)
per.fit(X_train,Y_train)

# print(Y_train.shape)
