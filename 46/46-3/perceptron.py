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

        # losses=[]
        fig=plt.figure(figsize=(10,10))
        ax=fig.add_subplot(projection='3d')
        xvalues = np.arange(X_train[:,0].min(), X_train[:,0].max())
        yvalues = np.arange(X_train[:,1].min(), X_train[:,1].max())
        xx, yy = np.meshgrid(xvalues, yvalues)
        for n in range(self.epochs):
            for i in range(self.X_train.shape[0]):
                x=self.X_train[i]
                y=self.Y_train[i]
                y_pred_train=x @ self.W + self.b
                
                error=y-y_pred_train

                self.W = self.W + error*x*self.learning_rate_w
                self.b = self.b + error*self.learning_rate_b

                # loss=np.mean(np.abs(error))
                # losses.append(loss)

                # Y_pred_train=self.X_train*self.W+self.b

                
            z = xx * (self.W)[0] + yy * (self.W)[1] + self.b # Y_pred
            ax.clear()
            ax.scatter(X_train[:,0],X_train[:,1],Y_train)
            ax.plot_surface(xx,yy,z,alpha=0.5, color="yellow")
            ax.set_xlabel("RM")
            ax.set_ylabel("ZN")
            ax.set_zlabel("price")
            plt.pause(0.1)


                # ax1.clear()
                # ax1.scatter(self.X_train,self.Y_train,color="blue")
                # ax1.plot(self.X_train,Y_pred_train, color="red")
                # ax1.set_title("perceptron method")
                # ax1.set_xlabel("Length")
                # ax1.set_ylabel("Height")

                # ax2.clear()
                # ax2.plot(losses)
                # ax2.set_title("Loss")
                # plt.pause(0.01)
        plt.pause(30)

        return self.W,self.b

    # def predict(self,X_test):
    #     Y_pred=X_test*self.W + self.b
        
    #     return Y_pred

    # def evaluate(self,X_test,Y_test):
    #     Y_pred=self.predict(X_test)
    #     MSE=np.mean(np.square(Y_test-Y_pred))
    #     plt.scatter(self.X_train,self.Y_train,color="blue")
    #     plt.plot(self.X_train,self.X_train*self.W+self.b, color="red")
    #     plt.scatter(X_test,Y_test,color="hotpink")
    #     plt.legend(['train data','training line','test data'])
    #     plt.xlabel("Length")
    #     plt.ylabel("Height")
    #     plt.show()
    #     return MSE