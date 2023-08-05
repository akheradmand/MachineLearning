import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self,W,b,lr_w,lr_b,epochs):
        self.W=W
        self.b=b
        self.learning_rate_w=lr_w
        self.learning_rate_b=lr_b
        self.epochs=epochs

    # training
    def fit(self,X_train,Y_train):
        self.X_train=X_train
        self.Y_train=Y_train

        losses=[]
        fig,(ax1,ax2)=plt.subplots(1,2)

        for n in range(self.epochs):
            for i in range(self.X_train.shape[0]):
                x=self.X_train[i]
                y=self.Y_train[i]
                y_pred=x*self.W + self.b
                
                error=y-y_pred

                self.W = self.W + error*x*self.learning_rate_w
                self.b = self.b + error*self.learning_rate_b

                loss=np.mean(np.abs(error))
                losses.append(loss)

                Y_pred=self.X_train*self.W+self.b

                ax1.clear()
                ax1.scatter(self.X_train,self.Y_train,color="blue")
                ax1.plot(self.X_train,Y_pred, color="red")

                ax2.clear()
                ax2.plot(losses)
                plt.pause(0.01)
        # plt.pause(5)
        # return ax1
        fig.clear()

    def predict(self,X_test):
        
        plt.scatter(self.X_train,self.Y_train,color="blue")
        plt.plot(self.X_train,self.X_train*self.W+self.b, color="red")
        Y_pred=X_test*self.W + self.b
        plt.plot(X_test,Y_pred,color="hotpink")
        plt.show()

    def evaluate(self):
        ...

    