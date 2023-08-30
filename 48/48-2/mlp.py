import numpy as np

class Mlp:
    def __init__(self,epochs,η,D_in,H1,H2,D_out):
        self.epochs=epochs
        self.η=η
        self.D_in=D_in
        self.H1=H1
        self.H2=H2
        self.D_out=D_out

        self.W1=np.random.randn(D_in,H1)
        self.W2=np.random.randn(H1,H2)
        self.W3=np.random.randn(H2,D_out)

        self.B1=np.random.randn(1,H1)
        self.B2=np.random.randn(1,H2)
        self.B3=np.random.randn(1,D_out)

    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def softmax(self,X):
        return np.exp(X)/np.sum(np.exp(X))

    def root_mean_square_error(self,Y_gt,Y_pred):
        return np.sqrt(np.mean((Y_gt-Y_pred)**2))

    def calculate_accuracy(self,Y,Y_pred):
        return np.sum(np.argmax(Y,axis=1)==np.argmax(Y_pred,axis=1))/len(Y)
    
    def forward(self,x):
        # layer1
        out1 = self.sigmoid(x.T @ self.W1 + self.B1)

        # layer2
        out2 = self.sigmoid(out1 @ self.W2 + self.B2)

        # layer3
        out3 = self.softmax(out2 @ self.W3 + self.B3)
        
        return out1,out2,out3

    def backward(self,x,y,y_pred,out1,out2,out3):
        # layer3
        # گرادیان = ارور * مشتق
        error = -2 * (y - y_pred)
        grad_B3 = error
        grad_W3 = out2.T @ error

        # layer2
        error = error @ self.W3.T * out2 *(1 - out2)
        grad_B2 = error
        grad_W2 = out1.T @ error

        # layer1
        error = error @ self.W2.T * out1 * (1 - out1)
        grad_B1 = error
        grad_W1 = x @ error

        return grad_B3,grad_W3,grad_B2,grad_W2,grad_B1,grad_W1

    def update(self,grad_B3,grad_W3,grad_B2,grad_W2,grad_B1,grad_W1):
        # layer1
        self.W1 -= self.η * grad_W1
        self.B1 -= self.η * grad_B1

        # layer2
        self.W2 -= self.η * grad_W2
        self.B2 -= self.η * grad_B2

        # layer3
        self.W3 -= self.η * grad_W3
        self.B3 -= self.η * grad_B3

    def fit(self,X_train,Y_train,X_test,Y_test):
        train_losses=[]
        train_accs=[]
        test_losses=[]
        test_accs=[]
        for epoch in range(self.epochs):
            # train
            Y_pred_train = []
            for x,y in zip(X_train,Y_train):

                x = x.reshape(-1,1)

                out1,out2,out3=self.forward(x)
                y_pred=out3
                Y_pred_train.append(y_pred)
                grad_B3,grad_W3,grad_B2,grad_W2,grad_B1,grad_W1=self.backward(x,y,y_pred,out1,out2,out3)
                self.update(grad_B3,grad_W3,grad_B2,grad_W2,grad_B1,grad_W1)

            # test
            Y_pred_test=[]
            for x,y in zip(X_test,Y_test):

                x = x.reshape(-1,1)

                out1,out2,out3=self.forward(x)
                y_pred = out3
                Y_pred_test.append(y_pred)
        

            loss_train,accuracy_train=self.evaluate(X_train,Y_train,Y_pred_train)
            loss_test,accuracy_test=self.evaluate(X_test,Y_test,Y_pred_test)

            train_losses.append(loss_train)
            train_accs.append(accuracy_train)
            test_losses.append(loss_test)
            test_accs.append(accuracy_test)
            print(f"{accuracy_train:^25}{loss_train:^25}{accuracy_test:^25}{loss_test:^25}")

        return train_losses,train_accs,test_losses,test_accs

    def predict(self,X):
      Y_pred=[]
      for x in X:
            x = x.reshape(-1,1)
            out1,out2,out3=self.forward(x)
            y_pred = out3
            Y_pred.append(y_pred)
      return Y_pred

    def evaluate(self,X,Y,Y_pred=None): # برای جلوگیری از محاسبه مجدد Y_pred
        if Y_pred==None:
            Y_pred=self.predict(X)

        Y_pred=np.array(Y_pred).reshape(-1,10)
        loss=self.root_mean_square_error(Y,Y_pred)
        accuracy=self.calculate_accuracy(Y,Y_pred)
        return loss,accuracy